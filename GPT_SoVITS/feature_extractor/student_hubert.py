"""Distilled HuBERT frontend (2.97M params, ~32x smaller than chinese-hubert-base).

Drop-in replacement for CNHubert.model: consumes raw 16k waveform (B, T) and
returns {"last_hidden_state": (B, T/320, 768)} in fp16 when requested.

Weights: pretrained_models/distilled_hubert/student_v0.pt
Distillation report: see upstream GPT-SoVITS distill_lab/REPORT_student.md
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained_models", "distilled_hubert", "student_v0.pt",
)


class _ConvBlock(nn.Module):
    def __init__(self, ci, co, k, s):
        super().__init__()
        self.conv = nn.Conv1d(ci, co, k, stride=s)
        self.norm = nn.GroupNorm(8, co)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _TCNBlock(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, dilation=d, padding=d)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, 3, dilation=d, padding=d)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return x + h


class StudentSSL(nn.Module):
    """7-layer conv frontend (320x downsample to 50Hz) + TCN + 2-layer attention."""

    def __init__(self, c=192, dilations=(1, 2, 4, 8, 16, 32), n_attn=2):
        super().__init__()
        self.front = nn.Sequential(
            _ConvBlock(1, c, 10, 5),
            _ConvBlock(c, c, 3, 2),
            _ConvBlock(c, c, 3, 2),
            _ConvBlock(c, c, 3, 2),
            _ConvBlock(c, c, 3, 2),
            _ConvBlock(c, c, 2, 2),
            _ConvBlock(c, c, 2, 2),
        )
        self.tcn = nn.Sequential(*[_TCNBlock(c, d) for d in dilations])
        layer = nn.TransformerEncoderLayer(
            c, nhead=4, dim_feedforward=4 * c, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=n_attn)
        self.head = nn.Conv1d(c, 768, 1)

    def forward(self, x):  # x: (B, T) raw waveform
        h = self.front(x.unsqueeze(1))
        h = self.tcn(h)
        h = self.attn(h.transpose(1, 2)).transpose(1, 2)
        return self.head(h)  # (B, 768, T/320)


class DistilledHubertFront(nn.Module):
    """Mimics the HuggingFace HubertModel interface used by the pipeline."""

    def __init__(self, c=192):
        super().__init__()
        self.student = StudentSSL(c=c)

    def forward(self, input_values, *args, **kwargs):
        dt = input_values.dtype
        # compute in fp32 (matches validated deployment), cast back for ssl_proj
        f = self.student(input_values.float())
        return {"last_hidden_state": f.transpose(1, 2).to(dt)}  # (B, T50, 768)


class DistilledCNHubert(nn.Module):
    """Stands in for CNHubert: exposes .model with the HF-style interface.
    Student weights stay fp32 (validated numerics); output dtype follows input."""

    def __init__(self, ckpt_path=DEFAULT_CKPT, device="cpu", dtype=torch.float32):
        super().__init__()
        front = DistilledHubertFront()
        state = torch.load(ckpt_path, map_location="cpu")
        front.student.load_state_dict(state["student"])
        front = front.eval().to(device)
        self.model = front
        self.eval_acc = state.get("eval_acc")
        self.params = state.get("params")

    def forward(self, x):
        return self.model(x)


def load_distilled_hubert(ckpt_path=DEFAULT_CKPT, device="cpu", is_half=False):
    m = DistilledCNHubert(ckpt_path, device=device)
    print(f"Distilled HuBERT loaded: {m.params/1e6:.2f}M params "
          f"(code acc vs teacher {m.eval_acc:.3f})")
    return m
