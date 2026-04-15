import warnings

warnings.filterwarnings("ignore")
import json
import math
import os
import time

import torch
from torch import nn
from torch.nn import functional as F

from module import commons
from module import modules
from module import attentions
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from module.commons import init_weights, get_padding
from module.mrte_model import MRTE
from module.quantize import ResidualVectorQuantizer

# from text import symbols
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2
from torch.cuda.amp import autocast
import contextlib

VITS_GENERATOR_BENCH_ENABLED = os.environ.get("GPTSOVITS_BENCH_VITS_GENERATOR", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VITS_GENERATOR_BENCH_DETAIL_ENABLED = os.environ.get("GPTSOVITS_BENCH_VITS_GENERATOR_DETAIL", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VITS_GENERATOR_BENCH_PREFIX = "GPTSOVITS_VITS_GENERATOR "


def _emit_vits_generator_bench(metrics: dict):
    payload = {}
    for key, value in metrics.items():
        payload[key] = round(value, 6) if isinstance(value, float) else value
    print(VITS_GENERATOR_BENCH_PREFIX + json.dumps(payload, ensure_ascii=False))


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


WINDOW = {}

class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        version="v2",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.version = version

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.encoder_text = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        if self.version == "v1":
            symbols = symbols_v1.symbols
        else:
            symbols = symbols_v2.symbols
        self.text_embedding = nn.Embedding(len(symbols), hidden_channels)

        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, text, text_lengths, ge, speed=1, test=None, result_length:int=None, overlap_frames:torch.Tensor=None, padding_length:int=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)

        y = self.ssl_proj(y * y_mask) * y_mask

        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.unsqueeze(commons.sequence_mask(text_lengths, text.size(1)), 1).to(y.dtype)
        if test == 1:
            text[:, :] = 0
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)

        if padding_length is not None and padding_length!=0:
            y = y[:, :, :-padding_length]
            y_mask = y_mask[:, :, :-padding_length]


        y = self.encoder2(y * y_mask, y_mask)

        if result_length is not None:
            y = y[:, :, -result_length:]
            y_mask = y_mask[:, :, -result_length:]

        if overlap_frames is not None:
            overlap_len = overlap_frames.shape[-1]
            window = WINDOW.get(overlap_len, None)
            if window is None:
                # WINDOW[overlap_len] = torch.hann_window(overlap_len*2, device=y.device, dtype=y.dtype)
                WINDOW[overlap_len] = torch.sin(torch.arange(overlap_len*2, device=y.device) * torch.pi / (overlap_len*2))
                window = WINDOW[overlap_len]


            window = window.to(y.device)
            y[:,:,:overlap_len] = (
                window[:overlap_len].view(1, 1, -1) * y[:,:,:overlap_len]
                + window[overlap_len:].view(1, 1, -1) * overlap_frames
            )
            
        y_ = y
        y_mask_ = y_mask



        if speed != 1:
            y = F.interpolate(y, size=int(y.shape[-1] / speed) + 1, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, y_, y_mask_

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        y = self.mrte(y, y_mask, refer, refer_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask


class WNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm = modules.LayerNorm(out_channels)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        out = self.proj(x) * x_mask
        out = self.norm(out)
        return out


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        is_bias=False,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_kernels_inv = 1.0 / self.num_kernels
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=is_bias)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        bench_enabled = VITS_GENERATOR_BENCH_ENABLED
        input_frames = int(x.size(-1))
        if bench_enabled:
            t_total0 = time.perf_counter()
            c_total0 = time.process_time()
            conv_pre_sec = cond_sec = upsample_sec = resblock_sec = post_sec = 0.0
            conv_pre_cpu_sec = cond_cpu_sec = upsample_cpu_sec = resblock_cpu_sec = post_cpu_sec = 0.0
            upsample_stage_sec = []
            upsample_stage_cpu_sec = []
            resblock_stage_sec = []
            resblock_stage_cpu_sec = []
            resblock_block_sec = []
            resblock_block_cpu_sec = []

            t_stage0 = time.perf_counter()
            c_stage0 = time.process_time()
        x = self.conv_pre(x)
        if bench_enabled:
            conv_pre_sec = time.perf_counter() - t_stage0
            conv_pre_cpu_sec = time.process_time() - c_stage0

        if g is not None:
            if bench_enabled:
                t_stage0 = time.perf_counter()
                c_stage0 = time.process_time()
            x = x + self.cond(g)
            if bench_enabled:
                cond_sec = time.perf_counter() - t_stage0
                cond_cpu_sec = time.process_time() - c_stage0

        for i in range(self.num_upsamples):
            if bench_enabled:
                t_stage0 = time.perf_counter()
                c_stage0 = time.process_time()
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            if bench_enabled:
                stage_wall = time.perf_counter() - t_stage0
                stage_cpu = time.process_time() - c_stage0
                upsample_sec += stage_wall
                upsample_cpu_sec += stage_cpu
                upsample_stage_sec.append(stage_wall)
                upsample_stage_cpu_sec.append(stage_cpu)

            if bench_enabled:
                t_stage0 = time.perf_counter()
                c_stage0 = time.process_time()
            resblock_offset = i * self.num_kernels
            block_wall_vals = []
            block_cpu_vals = []
            for j in range(self.num_kernels):
                block = self.resblocks[resblock_offset + j]
                if bench_enabled and VITS_GENERATOR_BENCH_DETAIL_ENABLED:
                    t_block0 = time.perf_counter()
                    c_block0 = time.process_time()
                if j == 0:
                    xs = block(x)
                else:
                    xs.add_(block(x))
                if bench_enabled and VITS_GENERATOR_BENCH_DETAIL_ENABLED:
                    block_wall_vals.append(time.perf_counter() - t_block0)
                    block_cpu_vals.append(time.process_time() - c_block0)
            x = xs.mul_(self.num_kernels_inv)
            if bench_enabled:
                stage_wall = time.perf_counter() - t_stage0
                stage_cpu = time.process_time() - c_stage0
                resblock_sec += stage_wall
                resblock_cpu_sec += stage_cpu
                resblock_stage_sec.append(stage_wall)
                resblock_stage_cpu_sec.append(stage_cpu)
                if VITS_GENERATOR_BENCH_DETAIL_ENABLED:
                    resblock_block_sec.append(block_wall_vals)
                    resblock_block_cpu_sec.append(block_cpu_vals)

        if bench_enabled:
            t_stage0 = time.perf_counter()
            c_stage0 = time.process_time()
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        if bench_enabled:
            post_sec = time.perf_counter() - t_stage0
            post_cpu_sec = time.process_time() - c_stage0
            _emit_vits_generator_bench(
                {
                    "input_frames": input_frames,
                    "output_frames": int(x.size(-1)),
                    "num_upsamples": self.num_upsamples,
                    "num_kernels": self.num_kernels,
                    "conv_pre_sec": conv_pre_sec,
                    "conv_pre_cpu_sec": conv_pre_cpu_sec,
                    "cond_sec": cond_sec,
                    "cond_cpu_sec": cond_cpu_sec,
                    "upsample_sec": upsample_sec,
                    "upsample_cpu_sec": upsample_cpu_sec,
                    "upsample_stage_sec": upsample_stage_sec,
                    "upsample_stage_cpu_sec": upsample_stage_cpu_sec,
                    "resblock_sec": resblock_sec,
                    "resblock_cpu_sec": resblock_cpu_sec,
                    "resblock_stage_sec": resblock_stage_sec,
                    "resblock_stage_cpu_sec": resblock_stage_cpu_sec,
                    "resblock_block_sec": resblock_block_sec if VITS_GENERATOR_BENCH_DETAIL_ENABLED else [],
                    "resblock_block_cpu_sec": resblock_block_cpu_sec if VITS_GENERATOR_BENCH_DETAIL_ENABLED else [],
                    "post_sec": post_sec,
                    "post_cpu_sec": post_cpu_sec,
                    "total_sec": time.perf_counter() - t_total0,
                    "total_cpu_sec": time.process_time() - c_total0,
                }
            )

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


v2pro_set = {"v2Pro", "v2ProPlus"}


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, version=None):
        super(MultiPeriodDiscriminator, self).__init__()
        if version in v2pro_set:
            periods = [2, 3, 5, 7, 11, 17, 23]
        else:
            periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0)).unsqueeze(-1)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        d = (
            torch.sum(x**2, 1, keepdim=True)
            + torch.sum(self.embedding.weight**2, 1)
            - 2 * torch.matmul(x, self.embedding.weight.T)
        )
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies


class Quantizer(torch.nn.Module):
    def __init__(self, embed_dim=512, n_code_groups=4, n_codes=160):
        super(Quantizer, self).__init__()
        assert embed_dim % n_code_groups == 0
        self.quantizer_modules = nn.ModuleList(
            [Quantizer_module(n_codes, embed_dim // n_code_groups) for _ in range(n_code_groups)]
        )
        self.n_code_groups = n_code_groups
        self.embed_dim = embed_dim

    def forward(self, xin):
        # B, C, T
        B, C, T = xin.shape
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, self.embed_dim)
        x = torch.split(x, self.embed_dim // self.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        for _x, m in zip(x, self.quantizer_modules):
            _z_q, _min_indicies = m(_x)
            z_q.append(_z_q)
            min_indicies.append(_min_indicies)  # B * T,
        z_q = torch.cat(z_q, -1).reshape(xin.shape)
        loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
        z_q = xin + (z_q - xin).detach()
        z_q = z_q.transpose(1, 2)
        codes = torch.stack(min_indicies, -1).reshape(B, T, self.n_code_groups)
        return z_q, loss, codes.transpose(1, 2)

    def embed(self, x):
        # idx: N, 4, T
        x = x.transpose(1, 2)
        x = torch.split(x, 1, 2)
        ret = []
        for q, embed in zip(x, self.quantizer_modules):
            q = embed.embedding(q.squeeze(-1))
            ret.append(q)
        ret = torch.cat(ret, -1)
        return ret.transpose(1, 2)  # N, C, T


class CodePredictor(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_q=8,
        dims=1024,
        ssl_dim=768,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.vq_proj = nn.Conv1d(ssl_dim, hidden_channels, 1)
        self.ref_enc = modules.MelStyleEncoder(ssl_dim, style_vector_dim=hidden_channels)

        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)

        self.out_proj = nn.Conv1d(hidden_channels, (n_q - 1) * dims, 1)
        self.n_q = n_q
        self.dims = dims

    def forward(self, x, x_mask, refer, codes, infer=False):
        x = x.detach()
        x = self.vq_proj(x * x_mask) * x_mask
        g = self.ref_enc(refer, x_mask)
        x = x + g
        x = self.encoder(x * x_mask, x_mask)
        x = self.out_proj(x * x_mask) * x_mask
        logits = x.reshape(x.shape[0], self.n_q - 1, self.dims, x.shape[-1]).transpose(2, 3)
        target = codes[1:].transpose(0, 1)
        if not infer:
            logits = logits.reshape(-1, self.dims)
            target = target.reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits, target)
            return loss
        else:
            _, top10_preds = torch.topk(logits, 10, dim=-1)
            correct_top10 = torch.any(top10_preds == target.unsqueeze(-1), dim=-1)
            top3_acc = 100 * torch.mean(correct_top10.float()).detach().cpu().item()

            print("Top-10 Accuracy:", top3_acc, "%")

            pred_codes = torch.argmax(logits, dim=-1)
            acc = 100 * torch.mean((pred_codes == target).float()).detach().cpu().item()
            print("Top-1 Accuracy:", acc, "%")

            return pred_codes.transpose(0, 1)


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v2",
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            version=version,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # self.version=os.environ.get("version","v1")
        if self.version == "v1":
            self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)
        else:
            self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

        self.is_v2pro = self.version in v2pro_set
        if self.is_v2pro:
            self.sv_emb = nn.Linear(20480, gin_channels)
            self.ge_to512 = nn.Linear(gin_channels, 512)
            self.prelu = nn.PReLU(num_parameters=gin_channels)

    def forward(self, ssl, y, y_lengths, text, text_lengths, sv_emb=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
        if self.is_v2pro:
            sv_emb = self.sv_emb(sv_emb)  # B*20480->B*512
            ge += sv_emb.unsqueeze(-1)
            ge = self.prelu(ge)
            ge512 = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        with autocast(enabled=False):
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
            ssl = self.ssl_proj(ssl)
            quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])

        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge512 if self.is_v2pro else ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=ge)
        return (
            o,
            commit_loss,
            ids_slice,
            y_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, ssl, y, y_lengths, text, text_lengths, test=None, noise_scale=0.5):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)

        ssl = self.ssl_proj(ssl)
        quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge, test=test)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_mask, (z, z_p, m_p, logs_p)

    def build_decode_condition(self, refer, sv_emb=None):
        def get_ge(single_refer, single_sv_emb):
            ge = None
            if single_refer is not None:
                refer_lengths = torch.LongTensor([single_refer.size(2)]).to(single_refer.device)
                refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, single_refer.size(2)), 1).to(
                    single_refer.dtype
                )
                if self.version == "v1":
                    ge = self.ref_enc(single_refer * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(single_refer[:, :704] * refer_mask, refer_mask)
                if self.is_v2pro:
                    single_sv_emb = self.sv_emb(single_sv_emb)  # B*20480->B*512
                    ge += single_sv_emb.unsqueeze(-1)
                    ge = self.prelu(ge)
            return ge

        if type(refer) == list:
            ges = []
            for idx, _refer in enumerate(refer):
                ge = get_ge(_refer, sv_emb[idx] if self.is_v2pro else None)
                ges.append(ge)
            ge = torch.stack(ges, 0).mean(0)
        else:
            ge = get_ge(refer, sv_emb)

        ge_text = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro and ge is not None else ge
        return ge, ge_text

    def _expand_decode_condition_batch(self, cond, batch_size, name):
        if cond is None:
            return None
        if cond.size(0) == batch_size:
            return cond
        if cond.size(0) != 1:
            raise ValueError(f"{name} batch size mismatch: expected 1 or {batch_size}, got {cond.size(0)}")
        return cond.expand(batch_size, -1, -1)

    def _prepare_decode_lengths(self, codes, text, code_lengths=None, text_lengths=None):
        if codes.ndim != 3:
            raise ValueError(f"codes shape mismatch: expected 3 dims [n_q, B, T], got {tuple(codes.shape)}")
        batch_size = int(codes.size(1))
        if int(text.size(0)) != batch_size:
            raise ValueError(f"text batch size mismatch: expected {batch_size}, got {text.size(0)}")

        if code_lengths is None:
            code_lengths = torch.full((batch_size,), int(codes.size(2)), device=codes.device, dtype=torch.long)
        else:
            code_lengths = torch.as_tensor(code_lengths, device=codes.device, dtype=torch.long)
            if code_lengths.ndim != 1 or int(code_lengths.numel()) != batch_size:
                raise ValueError(f"code_lengths shape mismatch: expected ({batch_size},), got {tuple(code_lengths.shape)}")

        if text_lengths is None:
            text_lengths = torch.full((batch_size,), int(text.size(1)), device=text.device, dtype=torch.long)
        else:
            text_lengths = torch.as_tensor(text_lengths, device=text.device, dtype=torch.long)
            if text_lengths.ndim != 1 or int(text_lengths.numel()) != batch_size:
                raise ValueError(f"text_lengths shape mismatch: expected ({batch_size},), got {tuple(text_lengths.shape)}")

        y_lengths = code_lengths * 2
        return y_lengths, text_lengths

    def _sample_decode_noise_like(self, target, y_lengths, sequential=False):
        if not sequential:
            return torch.randn_like(target)

        noise = torch.zeros_like(target)
        for idx, length in enumerate(y_lengths.tolist()):
            valid_len = int(length)
            if valid_len <= 0:
                continue
            noise[idx, :, :valid_len] = torch.randn(
                (target.size(1), valid_len),
                device=target.device,
                dtype=target.dtype,
            )
        return noise

    @torch.no_grad()
    def prepare_decode_latent(
        self,
        codes,
        text,
        refer,
        noise_scale=0.5,
        speed=1,
        sv_emb=None,
        ge=None,
        ge_text=None,
        code_lengths=None,
        text_lengths=None,
        sequential_noise=False,
    ):
        if ge is None:
            ge, ge_text = self.build_decode_condition(refer, sv_emb)
        elif ge_text is None:
            ge_text = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge

        batch_size = int(codes.size(1))
        ge = self._expand_decode_condition_batch(ge, batch_size, "ge")
        ge_text = self._expand_decode_condition_batch(ge_text, batch_size, "ge_text")
        y_lengths, text_lengths = self._prepare_decode_lengths(
            codes,
            text,
            code_lengths=code_lengths,
            text_lengths=text_lengths,
        )

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        x, m_p, logs_p, y_mask, _, _ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            ge_text,
            speed,
        )
        noise = self._sample_decode_noise_like(m_p, y_lengths, sequential=sequential_noise)
        z_p = m_p + noise * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        return z, y_mask, ge, ge_text, y_lengths, text_lengths

    @torch.no_grad()
    def decode(
        self,
        codes,
        text,
        refer,
        noise_scale=0.5,
        speed=1,
        sv_emb=None,
        ge=None,
        ge_text=None,
        code_lengths=None,
        text_lengths=None,
    ):
        z, y_mask, ge, _, _, _ = self.prepare_decode_latent(
            codes,
            text,
            refer,
            noise_scale=noise_scale,
            speed=speed,
            sv_emb=sv_emb,
            ge=ge,
            ge_text=ge_text,
            code_lengths=code_lengths,
            text_lengths=text_lengths,
            sequential_noise=False,
        )

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o


    @torch.no_grad()
    def decode_streaming(
        self,
        codes,
        text,
        refer,
        noise_scale=0.5,
        speed=1,
        sv_emb=None,
        result_length:int=None,
        overlap_frames:torch.Tensor=None,
        padding_length:int=None,
        ge=None,
        ge_text=None,
        code_lengths=None,
        text_lengths=None,
    ):
        if ge is None:
            ge, ge_text = self.build_decode_condition(refer, sv_emb)
        elif ge_text is None:
            ge_text = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge

        batch_size = int(codes.size(1))
        ge = self._expand_decode_condition_batch(ge, batch_size, "ge")
        ge_text = self._expand_decode_condition_batch(ge_text, batch_size, "ge_text")
        y_lengths, text_lengths = self._prepare_decode_lengths(
            codes,
            text,
            code_lengths=code_lengths,
            text_lengths=text_lengths,
        )

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
            result_length = (2*result_length) if result_length is not None else None
            padding_length = (2*padding_length) if padding_length is not None else None
        x, m_p, logs_p, y_mask, y_, y_mask_ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            ge_text,
            speed,
            result_length=result_length, 
            overlap_frames=overlap_frames, 
            padding_length=padding_length
            )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_, y_mask_

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)


def set_no_grad(net_g):
    for name, param in net_g.named_parameters():
        param.requires_grad = False
