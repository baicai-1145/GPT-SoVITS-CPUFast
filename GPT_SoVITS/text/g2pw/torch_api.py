import math
import os
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as ao_quant

from .onnx_api import _G2PWBaseOnnxConverter, _find_first_existing_file

warnings.filterwarnings("ignore")


class Int8Embedding(nn.Module):
    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)

    @property
    def weight(self):
        return self.weight_int8.float() * self.scale

    def forward(self, input):
        return self.weight_int8[input].float() * self.scale[input]

    @staticmethod
    def from_float(embedding: nn.Embedding) -> "Int8Embedding":
        w = embedding.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp(min=1e-8)
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        return Int8Embedding(w_int8, scale)


class Int8Linear(nn.Module):
    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight_int8.float() * self.scale, self.bias)

    @staticmethod
    def from_float(linear: nn.Linear) -> "Int8Linear":
        w = linear.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp(min=1e-8)
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        return Int8Linear(w_int8, scale, linear.bias.data if linear.bias is not None else None)


@torch.fx.wrap
def _batch_gather(hidden, position_ids):
    batch_size = hidden.size(0)
    seq_len = hidden.size(1)
    flat = hidden.reshape(batch_size * seq_len, -1)
    offsets = torch.arange(batch_size, device=hidden.device) * seq_len
    return flat[position_ids + offsets]


@torch.fx.wrap
def _get_position_embeds(position_embeddings, seq_len):
    return position_embeddings(torch.arange(seq_len, device=position_embeddings.weight.device))


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position, type_vocab_size=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)
        x = self.word_embeddings(input_ids) + _get_position_embeds(self.position_embeddings, seq_len).unsqueeze(0) + self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(x)


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.quant = ao_quant.QuantStub()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dequant_q = ao_quant.DeQuantStub()
        self.dequant_k = ao_quant.DeQuantStub()
        self.dequant_v = ao_quant.DeQuantStub()

    def forward(self, x, attention_mask):
        B, S, _ = x.shape
        x_q = self.quant(x)
        q = self.dequant_q(self.query(x_q)).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.dequant_k(self.key(x_q)).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.dequant_v(self.value(x_q)).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return out


class BertAttentionOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.quant = ao_quant.QuantStub()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dequant = ao_quant.DeQuantStub()
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dequant(self.dense(self.quant(hidden_states))) + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_heads)
        self.output = BertAttentionOutput(hidden_size)

    def forward(self, x, attention_mask):
        self_out = self.self(x, attention_mask)
        return self.output(self_out, x)


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.quant = ao_quant.QuantStub()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.dequant = ao_quant.DeQuantStub()

    def forward(self, x):
        return F.gelu(self.dequant(self.dense(self.quant(x))))


class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.quant = ao_quant.QuantStub()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dequant = ao_quant.DeQuantStub()
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dequant(self.dense(self.quant(hidden_states))) + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_heads)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size)

    def forward(self, x, attention_mask):
        attn_out = self.attention(x, attention_mask)
        inter_out = self.intermediate(attn_out)
        return self.output(inter_out, attn_out)


class BertEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)])

    def forward(self, x, attention_mask):
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class BertModel_(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position=512):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position)
        self.encoder = BertEncoder(num_layers, hidden_size, num_heads, intermediate_size)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)
        extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        return self.encoder(x, extended_mask)


@torch.fx.wrap
def _compute_mask(descriptor_bias, char_descriptor, second_order_descriptor, num_pos, char_ids, pos_pred, phoneme_mask):
    mask_logits = (
        descriptor_bias(torch.zeros_like(char_ids))
        + char_descriptor(char_ids)
        + second_order_descriptor(char_ids * num_pos + pos_pred)
    )
    return torch.sigmoid(mask_logits) * phoneme_mask


class G2PWModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, num_labels, num_chars, num_pos=11):
        super().__init__()
        self.bert = BertModel_(vocab_size, hidden_size, num_layers, num_heads, intermediate_size)
        self.head_quant = ao_quant.QuantStub()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pos_classifier = nn.Linear(hidden_size, num_pos)
        self.head_dequant_cls = ao_quant.DeQuantStub()
        self.head_dequant_pos = ao_quant.DeQuantStub()
        self.descriptor_bias = nn.Embedding(1, num_labels)
        self.char_descriptor = nn.Embedding(num_chars, num_labels)
        self.second_order_descriptor = nn.Embedding(num_chars * num_pos, num_labels)
        self.num_pos = num_pos

    def forward(self, input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids):
        hidden = self.bert(input_ids, token_type_ids, attention_mask)
        h = _batch_gather(hidden, position_ids)

        h_q = self.head_quant(h)
        pos_pred = self.head_dequant_pos(self.pos_classifier(h_q)).argmax(dim=1)
        mask_weight = _compute_mask(
            self.descriptor_bias, self.char_descriptor, self.second_order_descriptor,
            self.num_pos, char_ids, pos_pred, phoneme_mask,
        )

        logits = self.head_dequant_cls(self.classifier(h_q))
        logits_max = logits.max(dim=1, keepdim=True).values
        exp_logits = torch.exp(logits - logits_max) * mask_weight
        probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        return probs


def _load_g2pw_model(pth_path: str) -> G2PWModel:
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    hidden_size = state_dict["bert.embeddings.word_embeddings.weight"].shape[1]
    vocab_size = state_dict["bert.embeddings.word_embeddings.weight"].shape[0]
    num_layers = max(int(k.split(".")[3]) for k in state_dict if k.startswith("bert.encoder.layer.")) + 1
    intermediate_size = state_dict["bert.encoder.layer.0.intermediate.dense.weight"].shape[0]
    num_labels = state_dict["classifier.weight"].shape[0]
    num_chars = state_dict["char_descriptor.weight"].shape[0]

    model = G2PWModel(vocab_size, hidden_size, num_layers, hidden_size // 64, intermediate_size, num_labels, num_chars)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("bert.pooler") and k != "bert.embeddings.position_ids"}
    model.load_state_dict(filtered, strict=False)
    del state_dict, filtered
    model.eval()
    return model


FP32_GROUPS = [
    "bert.encoder.layer.0.output",
    "bert.encoder.layer.1.output",
    "bert.encoder.layer.2.output",
    "bert.encoder.layer.3.attention.self",
    "bert.encoder.layer.3.output",
    "bert.encoder.layer.4.output",
    "bert.encoder.layer.5.output",
    "bert.encoder.layer.6.output",
    "bert.encoder.layer.7.attention.self",
    "bert.encoder.layer.7.output",
    "bert.encoder.layer.8.output",
]


def _load_g2pw_int8_model(int8_path: str) -> G2PWModel:
    from torch.ao.nn.quantized import Linear as QLinear
    sd = torch.load(int8_path, map_location="cpu", weights_only=True)

    hidden_size = sd["bert.embeddings.word_embeddings.weight_int8"].shape[1]
    vocab_size = sd["bert.embeddings.word_embeddings.weight_int8"].shape[0]
    num_layers = max(int(k.split(".")[3]) for k in sd if k.startswith("bert.encoder.layer.")) + 1
    num_chars = sd["char_descriptor.weight_int8"].shape[0]
    for i in range(num_layers):
        key = f"bert.encoder.layer.{i}.intermediate.dense.weight"
        if key in sd:
            intermediate_size = sd[key].shape[0]
            break
    num_labels = sd["classifier.weight"].shape[0]

    def _pop(key):
        return sd.pop(key)

    def _make_qlinear(prefix):
        w_int8 = _pop(f"{prefix}.weight")
        s = _pop(f"{prefix}.w_scale")
        zp = _pop(f"{prefix}.w_zero_point")
        bias = sd.pop(f"{prefix}.bias", None)
        out_s = _pop(f"{prefix}.out_scale").item()
        out_zp = _pop(f"{prefix}.out_zero_point").item()
        if s.dim() == 0:
            qw = torch._make_per_tensor_quantized_tensor(w_int8, s.item(), zp.item())
        else:
            qw = torch._make_per_channel_quantized_tensor(w_int8, s, zp, 0)
        ql = QLinear(w_int8.shape[1], w_int8.shape[0])
        ql.set_weight_bias(qw, bias)
        ql.scale = out_s
        ql.zero_point = out_zp
        return ql

    def _make_int8emb(prefix):
        return Int8Embedding(_pop(f"{prefix}.weight_int8"), _pop(f"{prefix}.scale"))

    def _make_int8linear(prefix):
        return Int8Linear(_pop(f"{prefix}.weight_int8"), _pop(f"{prefix}.scale"), sd.pop(f"{prefix}.bias", None))

    def _make_linear(prefix):
        # 兼容: 新格式用 Int8Linear, 旧格式用 nn.Linear
        if f"{prefix}.weight_int8" in sd:
            return _make_int8linear(prefix)
        w = _pop(f"{prefix}.weight")
        b = sd.pop(f"{prefix}.bias", None)
        lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        lin.weight.data = w
        if b is not None:
            lin.bias.data = b
        return lin

    def _make_layernorm(prefix):
        w = _pop(f"{prefix}.weight")
        b = _pop(f"{prefix}.bias")
        ln = nn.LayerNorm(w.shape[0])
        ln.weight.data = w
        ln.bias.data = b
        return ln

    def _make_quantizer(prefix):
        q = torch.ao.nn.quantized.Quantize(
            _pop(f"{prefix}.scale").item(), _pop(f"{prefix}.zero_point").item(), torch.quint8,
        )
        return q

    def _make_dequantizer():
        return torch.ao.nn.quantized.DeQuantize()

    def _is_quantized(prefix):
        return f"{prefix}.out_scale" in sd

    def _build_self_attn(prefix):
        sa = BertSelfAttention.__new__(BertSelfAttention)
        nn.Module.__init__(sa)
        sa.num_heads = hidden_size // 64
        sa.head_dim = 64
        if _is_quantized(f"{prefix}.query"):
            sa.quant = _make_quantizer(f"{prefix}.quant")
            sa.query = _make_qlinear(f"{prefix}.query")
            sa.key = _make_qlinear(f"{prefix}.key")
            sa.value = _make_qlinear(f"{prefix}.value")
            sa.dequant_q = _make_dequantizer()
            sa.dequant_k = _make_dequantizer()
            sa.dequant_v = _make_dequantizer()
        else:
            sa.quant = ao_quant.QuantStub()
            sa.query = _make_linear(f"{prefix}.query")
            sa.key = _make_linear(f"{prefix}.key")
            sa.value = _make_linear(f"{prefix}.value")
            sa.dequant_q = ao_quant.DeQuantStub()
            sa.dequant_k = ao_quant.DeQuantStub()
            sa.dequant_v = ao_quant.DeQuantStub()
        return sa

    def _build_submodule(prefix, cls, has_residual=False):
        mod = cls.__new__(cls)
        nn.Module.__init__(mod)
        if _is_quantized(f"{prefix}.dense"):
            mod.quant = _make_quantizer(f"{prefix}.quant")
            mod.dense = _make_qlinear(f"{prefix}.dense")
            mod.dequant = _make_dequantizer()
        else:
            mod.quant = ao_quant.QuantStub()
            mod.dense = _make_linear(f"{prefix}.dense")
            mod.dequant = ao_quant.DeQuantStub()
        if has_residual:
            mod.LayerNorm = _make_layernorm(f"{prefix}.LayerNorm")
        return mod

    # Build model
    model = G2PWModel.__new__(G2PWModel)
    nn.Module.__init__(model)
    model.num_pos = 11

    bert = BertModel_.__new__(BertModel_)
    nn.Module.__init__(bert)

    emb = BertEmbeddings.__new__(BertEmbeddings)
    nn.Module.__init__(emb)
    emb.word_embeddings = _make_int8emb("bert.embeddings.word_embeddings")
    emb.position_embeddings = _make_int8emb("bert.embeddings.position_embeddings")
    emb.token_type_embeddings = _make_int8emb("bert.embeddings.token_type_embeddings")
    emb.LayerNorm = _make_layernorm("bert.embeddings.LayerNorm")
    bert.embeddings = emb

    encoder = BertEncoder.__new__(BertEncoder)
    nn.Module.__init__(encoder)
    layers = []
    for i in range(num_layers):
        lp = f"bert.encoder.layer.{i}"
        layer = BertLayer.__new__(BertLayer)
        nn.Module.__init__(layer)

        attn = BertAttention.__new__(BertAttention)
        nn.Module.__init__(attn)
        attn.self = _build_self_attn(f"{lp}.attention.self")
        attn.output = _build_submodule(f"{lp}.attention.output", BertAttentionOutput, has_residual=True)
        layer.attention = attn
        layer.intermediate = _build_submodule(f"{lp}.intermediate", BertIntermediate)
        layer.output = _build_submodule(f"{lp}.output", BertOutput, has_residual=True)
        layers.append(layer)
    encoder.layer = nn.ModuleList(layers)
    bert.encoder = encoder
    model.bert = bert

    if _is_quantized("classifier"):
        model.head_quant = _make_quantizer("head_quant")
        model.classifier = _make_qlinear("classifier")
        model.pos_classifier = _make_qlinear("pos_classifier")
        model.head_dequant_cls = _make_dequantizer()
        model.head_dequant_pos = _make_dequantizer()
    else:
        model.head_quant = ao_quant.QuantStub()
        model.classifier = _make_linear("classifier")
        model.pos_classifier = _make_linear("pos_classifier")
        model.head_dequant_cls = ao_quant.DeQuantStub()
        model.head_dequant_pos = ao_quant.DeQuantStub()
    model.descriptor_bias = _make_int8emb("descriptor_bias")
    model.char_descriptor = _make_int8emb("char_descriptor")
    model.second_order_descriptor = _make_int8emb("second_order_descriptor")

    del sd
    model.eval()
    return model


class G2PWTorchConverter(_G2PWBaseOnnxConverter):
    def __init__(
        self,
        model_dir: str = "G2PWModel/",
        style: str = "bopomofo",
        model_source: str = None,
        enable_non_tradional_chinese: bool = False,
    ):
        super().__init__(
            model_dir=model_dir,
            style=style,
            model_source=model_source,
            enable_non_tradional_chinese=enable_non_tradional_chinese,
        )
        int8_path = os.path.join(self.model_dir, "g2pw_int8.pth")
        if os.path.exists(int8_path):
            self.model = _load_g2pw_int8_model(int8_path)
        else:
            pth_path = _find_first_existing_file(
                os.path.join(self.model_dir, "g2pw.pth"),
                os.path.join(self.model_dir, "g2pW.pth"),
            )
            self.model = _load_g2pw_model(pth_path)

    def _predict(self, model_input: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        with torch.no_grad():
            probs = self.model(
                input_ids=torch.from_numpy(model_input["input_ids"]),
                token_type_ids=torch.from_numpy(model_input["token_type_ids"]),
                attention_mask=torch.from_numpy(model_input["attention_masks"]),
                phoneme_mask=torch.from_numpy(model_input["phoneme_masks"]).float(),
                char_ids=torch.from_numpy(model_input["char_ids"]),
                position_ids=torch.from_numpy(model_input["position_ids"]),
            ).numpy()

        preds = np.argmax(probs, axis=1).tolist()
        confidences = [float(probs[i, p]) for i, p in enumerate(preds)]
        return [self.labels[p] for p in preds], confidences


def load_bert_large_int8(int8_path: str) -> BertModel_:
    """加载 BERT-large INT8 量化模型，返回手写 BertModel_"""
    from torch.ao.nn.quantized import Linear as QLinear
    sd = torch.load(int8_path, map_location="cpu", weights_only=True)

    hidden_size = sd["embeddings.word_embeddings.weight_int8"].shape[1]
    num_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layer.")) + 1
    num_heads = hidden_size // 64
    for i in range(num_layers):
        for key in [f"encoder.layer.{i}.intermediate.dense.weight",
                    f"encoder.layer.{i}.intermediate.dense.weight_int8"]:
            if key in sd:
                intermediate_size = sd[key].shape[0]
                break
        else:
            continue
        break

    def _pop(key):
        return sd.pop(key)

    def _is_quantized(prefix):
        return f"{prefix}.out_scale" in sd

    def _make_qlinear(prefix):
        w = _pop(f"{prefix}.weight"); s = _pop(f"{prefix}.w_scale"); zp = _pop(f"{prefix}.w_zero_point")
        b = sd.pop(f"{prefix}.bias", None)
        out_s = _pop(f"{prefix}.out_scale").item(); out_zp = _pop(f"{prefix}.out_zero_point").item()
        if s.dim() == 0:
            qw = torch._make_per_tensor_quantized_tensor(w, s.item(), zp.item())
        else:
            qw = torch._make_per_channel_quantized_tensor(w, s, zp, 0)
        ql = QLinear(w.shape[1], w.shape[0]); ql.set_weight_bias(qw, b)
        ql.scale = out_s; ql.zero_point = out_zp
        return ql

    def _make_int8emb(prefix):
        return Int8Embedding(_pop(f"{prefix}.weight_int8"), _pop(f"{prefix}.scale"))

    def _make_int8linear(prefix):
        return Int8Linear(_pop(f"{prefix}.weight_int8"), _pop(f"{prefix}.scale"), sd.pop(f"{prefix}.bias", None))

    def _make_linear(prefix):
        if f"{prefix}.weight_int8" in sd:
            return _make_int8linear(prefix)
        if _is_quantized(prefix):
            return _make_qlinear(prefix)
        w = _pop(f"{prefix}.weight"); b = sd.pop(f"{prefix}.bias", None)
        lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        lin.weight.data = w
        if b is not None:
            lin.bias.data = b
        return lin

    def _make_layernorm(prefix):
        w = _pop(f"{prefix}.weight"); b = _pop(f"{prefix}.bias")
        ln = nn.LayerNorm(w.shape[0]); ln.weight.data = w; ln.bias.data = b
        return ln

    def _make_quantizer(prefix):
        return torch.ao.nn.quantized.Quantize(
            _pop(f"{prefix}.scale").item(), _pop(f"{prefix}.zero_point").item(), torch.quint8)

    def _make_dequantizer():
        return torch.ao.nn.quantized.DeQuantize()

    def _build_self_attn(prefix):
        sa = BertSelfAttention.__new__(BertSelfAttention); nn.Module.__init__(sa)
        sa.num_heads = num_heads; sa.head_dim = 64
        if _is_quantized(f"{prefix}.query"):
            sa.quant = _make_quantizer(f"{prefix}.quant")
            sa.query = _make_qlinear(f"{prefix}.query")
            sa.key = _make_qlinear(f"{prefix}.key")
            sa.value = _make_qlinear(f"{prefix}.value")
            sa.dequant_q = _make_dequantizer(); sa.dequant_k = _make_dequantizer(); sa.dequant_v = _make_dequantizer()
        else:
            sa.quant = ao_quant.QuantStub()
            sa.query = _make_linear(f"{prefix}.query")
            sa.key = _make_linear(f"{prefix}.key")
            sa.value = _make_linear(f"{prefix}.value")
            sa.dequant_q = ao_quant.DeQuantStub(); sa.dequant_k = ao_quant.DeQuantStub(); sa.dequant_v = ao_quant.DeQuantStub()
        return sa

    def _build_submodule(prefix, cls, has_residual=False):
        mod = cls.__new__(cls); nn.Module.__init__(mod)
        if _is_quantized(f"{prefix}.dense"):
            mod.quant = _make_quantizer(f"{prefix}.quant")
            mod.dense = _make_qlinear(f"{prefix}.dense")
            mod.dequant = _make_dequantizer()
        else:
            mod.quant = ao_quant.QuantStub()
            mod.dense = _make_linear(f"{prefix}.dense")
            mod.dequant = ao_quant.DeQuantStub()
        if has_residual:
            mod.LayerNorm = _make_layernorm(f"{prefix}.LayerNorm")
        return mod

    model = BertModel_.__new__(BertModel_); nn.Module.__init__(model)
    emb = BertEmbeddings.__new__(BertEmbeddings); nn.Module.__init__(emb)
    emb.word_embeddings = _make_int8emb("embeddings.word_embeddings")
    emb.position_embeddings = _make_int8emb("embeddings.position_embeddings")
    emb.token_type_embeddings = _make_int8emb("embeddings.token_type_embeddings")
    emb.LayerNorm = _make_layernorm("embeddings.LayerNorm")
    model.embeddings = emb

    encoder = BertEncoder.__new__(BertEncoder); nn.Module.__init__(encoder)
    layers = []
    for i in range(num_layers):
        lp = f"encoder.layer.{i}"
        layer = BertLayer.__new__(BertLayer); nn.Module.__init__(layer)
        attn = BertAttention.__new__(BertAttention); nn.Module.__init__(attn)
        attn.self = _build_self_attn(f"{lp}.attention.self")
        attn.output = _build_submodule(f"{lp}.attention.output", BertAttentionOutput, has_residual=True)
        layer.attention = attn
        layer.intermediate = _build_submodule(f"{lp}.intermediate", BertIntermediate)
        layer.output = _build_submodule(f"{lp}.output", BertOutput, has_residual=True)
        layers.append(layer)
    encoder.layer = nn.ModuleList(layers)
    model.encoder = encoder

    del sd
    model.eval()
    return model


class LightTokenizer:
    """轻量 BERT tokenizer，替代 transformers.AutoTokenizer，省 238 MB"""
    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer as _Tokenizer
        self._tok = _Tokenizer.from_file(tokenizer_json_path)

    def __call__(self, text, return_tensors=None, **kwargs):
        import torch as _torch
        enc = self._tok.encode(text)
        out = {
            "input_ids": enc.ids,
            "token_type_ids": enc.type_ids,
            "attention_mask": enc.attention_mask,
        }
        if return_tensors == "pt":
            out = {k: _torch.tensor([v]) for k, v in out.items()}
        return out
