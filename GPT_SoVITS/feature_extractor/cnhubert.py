import json
import logging
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.getLogger("numba").setLevel(logging.WARNING)

cnhubert_base_path = None


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


ACT2FN = {
    "gelu": _gelu,
}


class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        in_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(hidden_states))


class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        in_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.GroupNorm(num_groups=out_dim, num_channels=out_dim, affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.activation(hidden_states)


class HubertLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        in_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states.transpose(-2, -1)).transpose(-2, -1)
        return self.activation(hidden_states)


class HubertFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feat_extract_norm == "group":
            conv_layers = [HubertGroupNormConvLayer(config, layer_id=0)] + [
                HubertNoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                HubertLayerNormConvLayer(config, layer_id=i)
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(f"Unsupported feat_extract_norm: {config.feat_extract_norm}")
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class HubertFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        return self.dropout(hidden_states)


class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings: int):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.conv = nn.utils.weight_norm(conv, name="weight", dim=2)
        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states.transpose(1, 2)


class HubertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.scaling = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class HubertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return self.output_dropout(hidden_states)


class HubertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = HubertAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(attn_residual + hidden_states)
        hidden_states = self.final_layer_norm(hidden_states + self.feed_forward(hidden_states))
        return hidden_states


class HubertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.pos_conv_embed(hidden_states).to(hidden_states.device)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class HubertInferenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.zeros(config.hidden_size))
        self.encoder = HubertEncoder(config)

    def forward(self, input_values: torch.Tensor):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.feature_projection(extract_features)
        hidden_states = self.encoder(hidden_states)
        return {"last_hidden_state": hidden_states}


class WaveformFeatureExtractor:
    def __init__(self, config_dict: dict):
        self.sampling_rate = int(config_dict["sampling_rate"])
        self.do_normalize = bool(config_dict["do_normalize"])
        self.padding_value = float(config_dict["padding_value"])

    def _to_tensor(self, raw_speech, device: torch.device) -> torch.Tensor:
        if isinstance(raw_speech, torch.Tensor):
            tensor = raw_speech.detach()
        else:
            tensor = torch.as_tensor(raw_speech)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError("CNHuBERT only supports mono waveform inputs shaped as [T] or [B, T]")

        tensor = tensor.to(device=device, dtype=torch.float32)
        if self.do_normalize:
            mean = tensor.mean(dim=-1, keepdim=True)
            var = tensor.var(dim=-1, keepdim=True, unbiased=False)
            tensor = (tensor - mean) / torch.sqrt(var + 1e-7)
        return tensor

    def __call__(self, raw_speech, return_tensors: str = "pt", sampling_rate: int = 16000):
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate={self.sampling_rate}, but received sampling_rate={sampling_rate}"
            )
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")

        device = raw_speech.device if isinstance(raw_speech, torch.Tensor) else torch.device("cpu")
        input_values = self._to_tensor(raw_speech, device)
        return SimpleNamespace(input_values=input_values)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_config(base_path: str):
    config_path = os.path.join(base_path, "config.json")
    config_dict = _load_json(config_path)
    config = SimpleNamespace(**config_dict)
    config.conv_pos_batch_norm = config_dict.get("conv_pos_batch_norm", False)
    return config


class CNHubert(nn.Module):
    def __init__(self, base_path: str = None):
        super().__init__()
        if base_path is None:
            base_path = cnhubert_base_path
        if not os.path.exists(base_path):
            raise FileNotFoundError(base_path)

        self.base_path = base_path
        config = _load_config(base_path)
        self.model = HubertInferenceModel(config)
        preprocessor_config = _load_json(os.path.join(base_path, "preprocessor_config.json"))
        self.feature_extractor = WaveformFeatureExtractor(preprocessor_config)

        state_dict = torch.load(os.path.join(base_path, "pytorch_model.bin"), map_location="cpu")
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Failed to load CNHuBERT weights cleanly. missing={missing_keys}, unexpected={unexpected_keys}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats


def get_model():
    model = CNHubert()
    model.eval()
    return model


def get_content(hmodel, wav_16k_tensor):
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1, 2)


if __name__ == "__main__":
    from tools.audio_utils import load_audio_mono

    model = get_model()
    src_path = "/Users/Shared/原音频2.wav"
    wav_16k_tensor = torch.from_numpy(load_audio_mono(src_path, 16000))
    feats = get_content(model, wav_16k_tensor)
    print(feats.shape)
