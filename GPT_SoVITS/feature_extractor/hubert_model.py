"""手写 HuBERT forward，支持 W8A8 Eager mode 量化。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as ao_quant


class Int8Conv1d(nn.Module):
    def __init__(self, weight_int8, scale, bias=None, stride=1, padding=0, groups=1):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.stride = stride if isinstance(stride, (list, tuple)) else (stride,)
        self.padding = padding if isinstance(padding, (list, tuple)) else (padding,)
        self.groups = groups

    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        return F.conv1d(x, w, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

    @staticmethod
    def from_float(conv):
        w = conv.weight.data.float()
        scale = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) / 127.0
        scale = scale.clamp(min=1e-8)
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        return Int8Conv1d(w_int8, scale, conv.bias.data if conv.bias is not None else None,
                          stride=conv.stride, padding=conv.padding, groups=conv.groups)


class HubertFeatureExtractor(nn.Module):
    """7 层 Conv1d 前端，key 名对齐 HuggingFace"""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        # layer 0: Conv1d(1,512,10,5) + GroupNorm + GELU
        layer0 = nn.Module()
        layer0.conv = nn.Conv1d(1, 512, 10, stride=5, bias=False)
        layer0.layer_norm = nn.GroupNorm(512, 512)
        self.conv_layers.append(layer0)
        # layer 1-6: Conv1d(512,512,k,2) + GELU
        for k in [3, 3, 3, 3, 2, 2]:
            layer = nn.Module()
            layer.conv = nn.Conv1d(512, 512, k, stride=2, bias=False)
            self.conv_layers.append(layer)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.conv_layers[0].layer_norm(self.conv_layers[0].conv(x)))
        for layer in self.conv_layers[1:]:
            x = F.gelu(layer.conv(x))
        return x.transpose(1, 2)


class HubertFeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512)
        self.quant = ao_quant.QuantStub()
        self.projection = nn.Linear(512, 768)
        self.dequant = ao_quant.DeQuantStub()

    def forward(self, x):
        x = self.layer_norm(x)
        return self.dequant(self.projection(self.quant(x)))


class HubertPositionalConv(nn.Module):
    """位置编码 Conv1d (和 HF HubertPositionalConvEmbedding 对齐)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(768, 768, 128, padding=128 // 2, groups=16)
        self.num_pad_remove = 1  # kernel_size 128 是偶数，需要去掉右边 1 个

    def forward(self, x):
        x_t = x.transpose(1, 2)  # (B, 768, S)
        x_t = self.conv(x_t)
        if self.num_pad_remove > 0:
            x_t = x_t[:, :, :-self.num_pad_remove]
        x_t = F.gelu(x_t)
        return x + x_t.transpose(1, 2)


class HubertAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.quant = ao_quant.QuantStub()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dequant_q = ao_quant.DeQuantStub()
        self.dequant_k = ao_quant.DeQuantStub()
        self.dequant_v = ao_quant.DeQuantStub()
        self.quant_out = ao_quant.QuantStub()
        self.dequant_out = ao_quant.DeQuantStub()

    def forward(self, x):
        B, S, _ = x.shape
        x_q = self.quant(x)
        q = self.dequant_q(self.q_proj(x_q)).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.dequant_k(self.k_proj(x_q)).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.dequant_v(self.v_proj(x_q)).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return self.dequant_out(self.out_proj(self.quant_out(out)))


class HubertFeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.quant = ao_quant.QuantStub()
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.dequant_inter = ao_quant.DeQuantStub()
        self.quant_out = ao_quant.QuantStub()
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.dequant_out = ao_quant.DeQuantStub()

    def forward(self, x):
        x = F.gelu(self.dequant_inter(self.intermediate_dense(self.quant(x))))
        return self.dequant_out(self.output_dense(self.quant_out(x)))


class HubertEncoderLayer(nn.Module):
    """Post-norm Transformer layer (和 HuggingFace HubertEncoderLayer 对齐)"""
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.attention = HubertAttention(hidden_size, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = HubertFeedForward(hidden_size, intermediate_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Post-norm: Attention → residual → LayerNorm
        x = self.layer_norm(x + self.attention(x))
        # Post-norm: FFN → residual → final_LayerNorm
        x = self.final_layer_norm(x + self.feed_forward(x))
        return x


class HubertEncoder(nn.Module):
    def __init__(self, num_layers=12, hidden_size=768, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.pos_conv_embed = HubertPositionalConv()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([
            HubertEncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.pos_conv_embed(x)
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class HubertModel_(nn.Module):
    """手写 HuBERT，key 名对齐 HuggingFace HubertModel"""
    def __init__(self, num_layers=12, hidden_size=768, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.feature_extractor = HubertFeatureExtractor()
        self.feature_projection = HubertFeatureProjection()
        self.encoder = HubertEncoder(num_layers, hidden_size, num_heads, intermediate_size)

    def forward(self, waveform):
        # waveform: (B, T) 16kHz
        features = self.feature_extractor(waveform)  # (B, S, 512)
        projected = self.feature_projection(features)  # (B, S, 768)
        encoded = self.encoder(projected)  # (B, S, 768)
        return encoded


class CNHubert_(nn.Module):
    """和原始 CNHubert 兼容的包装"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        return {"last_hidden_state": self.model(waveform)}
