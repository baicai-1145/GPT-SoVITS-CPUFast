import torch
import os
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

import torch.nn as nn

cnhubert_base_path = None


class CNHubert(nn.Module):
    def __init__(self, base_path: str = None):
        super().__init__()
        if base_path is None:
            base_path = cnhubert_base_path
        if not os.path.exists(base_path):
            raise FileNotFoundError(base_path)

        int8_path = os.path.join(base_path, "cnhubert_int8.pth")
        self._is_int8 = os.path.exists(int8_path)
        if self._is_int8:
            self.model = _load_int8_hubert(int8_path)
        else:
            from transformers import HubertModel
            self.model = HubertModel.from_pretrained(base_path, local_files_only=True)

    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.model(x)
        if isinstance(out, dict):
            return out["last_hidden_state"]
        return out

    def half(self):
        if self._is_int8:
            return self
        return super().half()

    def to(self, *args, **kwargs):
        if self._is_int8:
            return self
        return super().to(*args, **kwargs)


class _Int8HubertWrapper(nn.Module):
    """包装手写 HuBERT，提供和 HubertModel 相同的接口"""
    def __init__(self, hubert_model):
        super().__init__()
        self._model = hubert_model

    def forward(self, input_values, **kwargs):
        x = input_values
        if x.dtype != torch.float32:
            x = x.float()
        hidden = self._model(x.cpu())
        return {"last_hidden_state": hidden.to(input_values.device)}

    def half(self):
        return self

    def to(self, *args, **kwargs):
        return self


def _load_int8_hubert(int8_path):
    from .hubert_model import (
        HubertModel_, HubertFeatureExtractor, HubertFeatureProjection,
        HubertPositionalConv, HubertAttention, HubertFeedForward,
        HubertEncoderLayer, HubertEncoder, Int8Conv1d,
    )
    from text.g2pw.torch_api import Int8Linear
    import torch.ao.quantization as ao_quant
    from torch.ao.nn.quantized import Linear as QLinear

    sd = torch.load(int8_path, map_location="cpu", weights_only=True)

    def _pop(key):
        return sd.pop(key)

    def _is_q(prefix):
        return f"{prefix}.out_scale" in sd

    def _make_ql(prefix):
        w = _pop(f"{prefix}.weight"); s = _pop(f"{prefix}.w_scale"); zp = _pop(f"{prefix}.w_zero_point")
        b = sd.pop(f"{prefix}.bias", None)
        out_s = _pop(f"{prefix}.out_scale").item(); out_zp = _pop(f"{prefix}.out_zero_point").item()
        qw = torch._make_per_channel_quantized_tensor(w, s, zp, 0) if s.dim() > 0 else torch._make_per_tensor_quantized_tensor(w, s.item(), zp.item())
        ql = QLinear(w.shape[1], w.shape[0]); ql.set_weight_bias(qw, b)
        ql.scale = out_s; ql.zero_point = out_zp
        return ql

    def _make_il(prefix):
        return Int8Linear(_pop(f"{prefix}.weight_int8"), _pop(f"{prefix}.scale"), sd.pop(f"{prefix}.bias", None))

    def _make_ic(prefix):
        w = _pop(f"{prefix}.weight_int8"); sc = _pop(f"{prefix}.scale"); b = sd.pop(f"{prefix}.bias", None)
        stride = tuple(_pop(f"{prefix}.stride").tolist()); pad = tuple(_pop(f"{prefix}.padding").tolist())
        groups = _pop(f"{prefix}.groups").item()
        return Int8Conv1d(w, sc, b, stride=stride, padding=pad, groups=groups)

    def _make_lin(prefix):
        if f"{prefix}.weight_int8" in sd:
            return _make_il(prefix)
        if _is_q(prefix):
            return _make_ql(prefix)
        w = _pop(f"{prefix}.weight"); b = sd.pop(f"{prefix}.bias", None)
        l = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        l.weight.data = w
        if b is not None:
            l.bias.data = b
        return l

    def _make_ln(prefix):
        w = _pop(f"{prefix}.weight"); b = _pop(f"{prefix}.bias")
        l = nn.LayerNorm(w.shape[0]); l.weight.data = w; l.bias.data = b
        return l

    def _make_gn(prefix):
        w = _pop(f"{prefix}.weight"); b = _pop(f"{prefix}.bias")
        g = nn.GroupNorm(w.shape[0], w.shape[0]); g.weight.data = w; g.bias.data = b
        return g

    def _make_quant(prefix):
        return torch.ao.nn.quantized.Quantize(_pop(f"{prefix}.scale").item(), _pop(f"{prefix}.zero_point").item(), torch.quint8)

    def _make_dequant():
        return torch.ao.nn.quantized.DeQuantize()

    # feature_extractor
    fe = HubertFeatureExtractor.__new__(HubertFeatureExtractor); nn.Module.__init__(fe)
    fe.conv_layers = nn.ModuleList()
    layer0 = nn.Module()
    layer0.conv = _make_ic("feature_extractor.conv_layers.0.conv")
    layer0.layer_norm = _make_gn("feature_extractor.conv_layers.0.layer_norm")
    fe.conv_layers.append(layer0)
    for ci in range(1, 7):
        layer = nn.Module()
        layer.conv = _make_ic(f"feature_extractor.conv_layers.{ci}.conv")
        fe.conv_layers.append(layer)

    # feature_projection
    fp = HubertFeatureProjection.__new__(HubertFeatureProjection); nn.Module.__init__(fp)
    fp.layer_norm = _make_ln("feature_projection.layer_norm")
    if _is_q("feature_projection.projection"):
        fp.quant = _make_quant("feature_projection.quant")
        fp.projection = _make_ql("feature_projection.projection")
        fp.dequant = _make_dequant()
    else:
        fp.quant = ao_quant.QuantStub()
        fp.projection = _make_lin("feature_projection.projection")
        fp.dequant = ao_quant.DeQuantStub()

    # encoder
    enc = HubertEncoder.__new__(HubertEncoder); nn.Module.__init__(enc)
    enc.pos_conv_embed = HubertPositionalConv.__new__(HubertPositionalConv); nn.Module.__init__(enc.pos_conv_embed)
    enc.pos_conv_embed.conv = _make_ic("encoder.pos_conv_embed.conv")
    enc.pos_conv_embed.num_pad_remove = 1
    enc.layer_norm = _make_ln("encoder.layer_norm")
    layers = []
    for i in range(12):
        lp = f"encoder.layers.{i}"
        layer = HubertEncoderLayer.__new__(HubertEncoderLayer); nn.Module.__init__(layer)

        attn = HubertAttention.__new__(HubertAttention); nn.Module.__init__(attn)
        attn.num_heads = 12; attn.head_dim = 64
        ap = f"{lp}.attention"
        if _is_q(f"{ap}.q_proj"):
            attn.quant = _make_quant(f"{ap}.quant")
            attn.q_proj = _make_ql(f"{ap}.q_proj"); attn.k_proj = _make_ql(f"{ap}.k_proj"); attn.v_proj = _make_ql(f"{ap}.v_proj")
            attn.dequant_q = _make_dequant(); attn.dequant_k = _make_dequant(); attn.dequant_v = _make_dequant()
            attn.quant_out = _make_quant(f"{ap}.quant_out"); attn.out_proj = _make_ql(f"{ap}.out_proj"); attn.dequant_out = _make_dequant()
        else:
            attn.quant = ao_quant.QuantStub()
            attn.q_proj = _make_lin(f"{ap}.q_proj"); attn.k_proj = _make_lin(f"{ap}.k_proj"); attn.v_proj = _make_lin(f"{ap}.v_proj")
            attn.dequant_q = ao_quant.DeQuantStub(); attn.dequant_k = ao_quant.DeQuantStub(); attn.dequant_v = ao_quant.DeQuantStub()
            attn.quant_out = ao_quant.QuantStub(); attn.out_proj = _make_lin(f"{ap}.out_proj"); attn.dequant_out = ao_quant.DeQuantStub()
        layer.attention = attn
        layer.layer_norm = _make_ln(f"{lp}.layer_norm")

        ffn = HubertFeedForward.__new__(HubertFeedForward); nn.Module.__init__(ffn)
        fp2 = f"{lp}.feed_forward"
        if _is_q(f"{fp2}.intermediate_dense"):
            ffn.quant = _make_quant(f"{fp2}.quant")
            ffn.intermediate_dense = _make_ql(f"{fp2}.intermediate_dense"); ffn.dequant_inter = _make_dequant()
            ffn.quant_out = _make_quant(f"{fp2}.quant_out")
            ffn.output_dense = _make_ql(f"{fp2}.output_dense"); ffn.dequant_out = _make_dequant()
        else:
            ffn.quant = ao_quant.QuantStub()
            ffn.intermediate_dense = _make_lin(f"{fp2}.intermediate_dense"); ffn.dequant_inter = ao_quant.DeQuantStub()
            ffn.quant_out = ao_quant.QuantStub()
            ffn.output_dense = _make_lin(f"{fp2}.output_dense"); ffn.dequant_out = ao_quant.DeQuantStub()
        layer.feed_forward = ffn
        layer.final_layer_norm = _make_ln(f"{lp}.final_layer_norm")
        layers.append(layer)

    enc.layers = nn.ModuleList(layers)

    hubert = HubertModel_.__new__(HubertModel_); nn.Module.__init__(hubert)
    hubert.feature_extractor = fe
    hubert.feature_projection = fp
    hubert.encoder = enc
    hubert.eval()

    del sd
    return _Int8HubertWrapper(hubert)


def get_model():
    model = CNHubert()
    model.eval()
    return model


def get_content(hmodel, wav_16k_tensor):
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1, 2)
