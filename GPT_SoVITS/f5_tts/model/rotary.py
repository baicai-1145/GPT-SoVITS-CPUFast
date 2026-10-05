"""Minimal rotary position embedding helpers.

Replaces the tiny slice of `x-transformers` used by the F5-TTS DiT so the
CPUFast wheel does not need the full dependency. Semantics match
`x_transformers.x_transformers.RotaryEmbedding` / `apply_rotary_pos_emb`.
"""

import torch


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, use_xpos=False, scale_base=512, interpolation_factor=1.0, base=10000.0,
                 base_rescale_factor=1.0):
        super().__init__()
        if use_xpos:
            raise ValueError("xpos is not supported by the lightweight RotaryEmbedding")
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.interpolation_factor = interpolation_factor

    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    def forward(self, t, offset=0):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        freqs = torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
        freqs = freqs / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = freqs.reshape(*freqs.shape[:-2], -1)
        return freqs, 1.0


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if torch.is_tensor(scale) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
        if torch.is_tensor(scale):
            scale = scale.unsqueeze(1)

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim=-1).type(orig_dtype)
