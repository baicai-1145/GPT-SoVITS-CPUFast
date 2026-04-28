# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import math
import os
import subprocess
import time
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F
MulticlassAccuracy = None  # 推理时不需要，避免 torchmetrics 拉 transformers
from AR.models.utils import (
    dpo_loss,
    get_batch_logps,
    make_pad_mask,
    make_pad_mask_left,
    make_reject_y,
    sample,
    topk_sampling,
)
from AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from AR.modules.transformer import LayerNorm, TransformerEncoder, TransformerEncoderLayer

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}

MAX_AR_DECODE_STEPS = 1500


def _alloc_token_buffer(initial_tokens: torch.Tensor, max_decode_steps: int) -> torch.Tensor:
    batch_size = initial_tokens.shape[0]
    prefix_len = initial_tokens.shape[1]
    buffer = torch.empty(
        (batch_size, prefix_len + max_decode_steps),
        dtype=initial_tokens.dtype,
        device=initial_tokens.device,
    )
    if prefix_len > 0:
        buffer[:, :prefix_len] = initial_tokens
    return buffer


def _compact_token_buffer(buffer: torch.Tensor, index: torch.Tensor, used_length: int) -> torch.Tensor:
    compacted = buffer.new_empty((int(index.numel()), buffer.shape[1]))
    if used_length > 0:
        compacted[:, :used_length] = torch.index_select(buffer[:, :used_length], dim=0, index=index)
    return compacted


def _compact_cache_buffer(cache: torch.Tensor, index: torch.Tensor, used_length: int) -> torch.Tensor:
    compacted = cache.new_empty((int(index.numel()), cache.shape[1], cache.shape[2]))
    if used_length > 0:
        compacted[:, :used_length, :] = torch.index_select(cache[:, :used_length, :], dim=0, index=index)
    return compacted


def _compact_decode_attn_mask_full(mask: torch.Tensor, index: torch.Tensor, prompt_width: int) -> torch.Tensor:
    compacted = mask.new_zeros((int(index.numel()), mask.shape[1], mask.shape[2], mask.shape[3]))
    if prompt_width > 0:
        compacted[:, :, :, :prompt_width] = torch.index_select(mask[:, :, :, :prompt_width], dim=0, index=index)
    return compacted


def _move_token_buffer_row(buffer: torch.Tensor, dst: int, src: int, used_length: int) -> None:
    if dst == src:
        return
    if used_length > 0:
        buffer[dst, :used_length] = buffer[src, :used_length]


def _move_cache_buffer_row(cache: torch.Tensor, dst: int, src: int, used_length: int) -> None:
    if dst == src:
        return
    if used_length > 0:
        cache[dst, :used_length, :] = cache[src, :used_length, :]


def _move_mask_row(mask: torch.Tensor, dst: int, src: int) -> None:
    if dst == src:
        return
    mask[dst] = mask[src]


def _get_benchmark_probe(model) -> Optional[dict]:
    probe = getattr(model, "_benchmark_probe", None)
    if isinstance(probe, dict):
        return probe
    return None


def _probe_add(probe: Optional[dict], key: str, delta: float) -> None:
    if probe is not None:
        probe[key] = probe.get(key, 0.0) + float(delta)


def _probe_inc(probe: Optional[dict], key: str, delta: int = 1) -> None:
    if probe is not None:
        probe[key] = int(probe.get(key, 0)) + int(delta)


def _probe_append(probe: Optional[dict], key: str, value) -> None:
    if probe is not None:
        bucket = probe.get(key)
        if not isinstance(bucket, list):
            bucket = []
            probe[key] = bucket
        bucket.append(value)


def _get_process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
        rss_kb = int(result.stdout.strip() or "0")
        return rss_kb * 1024
    except Exception:
        return 0


def _probe_update_rss_peak(probe: Optional[dict], decode_idx: int = -1) -> int:
    if probe is None:
        return 0
    rss_bytes = _get_process_rss_bytes()
    if rss_bytes <= 0:
        return 0
    if "rss_start_bytes" not in probe:
        probe["rss_start_bytes"] = int(rss_bytes)
    probe["rss_end_bytes"] = int(rss_bytes)
    if rss_bytes > int(probe.get("rss_peak_bytes", 0)):
        probe["rss_peak_bytes"] = int(rss_bytes)
        probe["rss_peak_decode_idx"] = int(decode_idx)
    return rss_bytes


# @torch.jit.script ## 使用的话首次推理会非常慢，而且推理速度不稳定
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    if scale is None:
        scale_factor = torch.tensor(1 / math.sqrt(query.size(-1)))
    else:
        scale_factor = scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask, 0)
        else:
            attn_mask[attn_mask != float("-inf")] = 0
            attn_mask[attn_mask == float("-inf")] = 1
            attn_weight.masked_fill_(attn_mask, 0)

    return attn_weight @ value


@torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.b1 = b1
        self.b2 = b2
        self.w1_t = w1.transpose(0, 1).contiguous()
        self.w2_t = w2.transpose(0, 1).contiguous()
        self.output_dim: int = b2.shape[0]

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        rows = batch_size * seq_len
        x2d = x.reshape(rows, x.shape[2])
        if rows == 1:
            x2d = F.relu(F.linear(x2d, self.w1_t.transpose(0, 1), self.b1))
            x2d = F.linear(x2d, self.w2_t.transpose(0, 1), self.b2)
        else:
            x2d = torch.addmm(self.b1, x2d, self.w1_t)
            x2d = F.relu(x2d)
            x2d = torch.addmm(self.b2, x2d, self.w2_t)
        return x2d.view(batch_size, seq_len, self.output_dim)


@torch.jit.script
class T2SBlock:
    def __init__(
        self,
        num_heads,
        hidden_dim: int,
        mlp: T2SMLP,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        norm_eps1,
        norm_w2,
        norm_b2,
        norm_eps2,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_b = qkv_b
        self.qkv_w_t = qkv_w.transpose(0, 1).contiguous()
        self.qkv_out_dim: int = qkv_b.shape[0]
        self.out_b = out_b
        self.out_w_t = out_w.transpose(0, 1).contiguous()
        self.out_out_dim: int = out_b.shape[0]
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

        self.false = torch.tensor(False, dtype=torch.bool)

    @torch.jit.ignore
    def to_mask(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ):
        if padding_mask is None:
            return x

        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        max_decode_steps: int,
        padding_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True,
    ):
        x_masked = self.to_mask(x, padding_mask)
        batch_size = x_masked.shape[0]
        q_len = x_masked.shape[1]
        rows = batch_size * q_len
        if rows == 1:
            qkv = F.linear(x_masked.reshape(rows, x_masked.shape[2]), self.qkv_w_t.transpose(0, 1), self.qkv_b)
        else:
            qkv = torch.addmm(self.qkv_b, x_masked.reshape(rows, x_masked.shape[2]), self.qkv_w_t)
        qkv = qkv.view(batch_size, q_len, self.qkv_out_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        kv_len = k.shape[1]

        q = self.to_mask(q, padding_mask)
        k_cache = self.to_mask(k, padding_mask)
        v_cache = self.to_mask(v, padding_mask)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn_masked = self.to_mask(attn, padding_mask)
        out_rows = batch_size * q_len
        if out_rows == 1:
            attn = F.linear(
                attn_masked.reshape(out_rows, attn_masked.shape[2]),
                self.out_w_t.transpose(0, 1),
                self.out_b,
            )
        else:
            attn = torch.addmm(self.out_b, attn_masked.reshape(out_rows, attn_masked.shape[2]), self.out_w_t)
        attn = attn.view(batch_size, q_len, self.out_out_dim)

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        cache_capacity = kv_len + max_decode_steps
        full_k_cache = k_cache.new_empty((batch_size, cache_capacity, k_cache.shape[2]))
        full_v_cache = v_cache.new_empty((batch_size, cache_capacity, v_cache.shape[2]))
        full_k_cache[:, :kv_len] = k_cache
        full_v_cache[:, :kv_len] = v_cache
        return x, full_k_cache, full_v_cache, kv_len

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_len: int,
        attn_mask: torch.Tensor = None,
        torch_sdpa: bool = True,
    ):
        batch_size = x.shape[0]
        q_len = x.shape[1]
        rows = batch_size * q_len
        if rows == 1:
            qkv = F.linear(x.reshape(rows, x.shape[2]), self.qkv_w_t.transpose(0, 1), self.qkv_b)
        else:
            qkv = torch.addmm(self.qkv_b, x.reshape(rows, x.shape[2]), self.qkv_w_t)
        qkv = qkv.view(batch_size, q_len, self.qkv_out_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        next_cache_len = cache_len + k.shape[1]
        k_cache[:, cache_len:next_cache_len] = k
        v_cache[:, cache_len:next_cache_len] = v

        kv_len = next_cache_len
        active_k_cache = k_cache[:, :kv_len]
        active_v_cache = v_cache[:, :kv_len]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = active_k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = active_v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, (~attn_mask) if attn_mask is not None else None)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        out_rows = batch_size * q_len
        if out_rows == 1:
            attn = F.linear(attn.reshape(out_rows, attn.shape[2]), self.out_w_t.transpose(0, 1), self.out_b)
        else:
            attn = torch.addmm(self.out_b, attn.reshape(out_rows, attn.shape[2]), self.out_w_t)
        attn = attn.view(batch_size, q_len, self.out_out_dim)

        x = x + attn
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w1,
            self.norm_b1,
            self.norm_eps1,
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks: int, blocks: List[T2SBlock]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        max_decode_steps: int,
        padding_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True,
    ):
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        cache_len: int = 0
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_, cache_len = self.blocks[i].process_prompt(
                x, attn_mask, max_decode_steps, padding_mask, torch_sdpa
            )
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache, cache_len

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        cache_len: int,
        attn_mask: torch.Tensor = None,
        torch_sdpa: bool = True,
    ):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(
                x, k_cache[i], v_cache[i], cache_len, attn_mask, torch_sdpa
            )
        return x, k_cache, v_cache, cache_len + 1


class Text2SemanticDecoder(nn.Module):
    def __init__(
        self,
        config,
        norm_first=False,
        top_k=3,
        build_t2s_transformer: bool = True,
        build_h_module: bool = True,
    ):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim,
            self.phoneme_vocab_size,
            self.p_dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim,
            self.vocab_size,
            self.p_dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        if build_h_module:
            self.h = TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=self.model_dim,
                    nhead=self.num_head,
                    dim_feedforward=self.model_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                ),
                num_layers=self.num_layers,
                norm=LayerNorm(self.model_dim) if norm_first else None,
            )
        else:
            self.h = None

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        ) if MulticlassAccuracy is not None else None

        if build_t2s_transformer:
            self.rebuild_t2s_transformer()

    def rebuild_t2s_transformer(self):
        self._require_h_transformer()
        blocks = []

        for i in range(self.num_layers):
            layer = self.h.layers[i]
            t2smlp = T2SMLP(
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias,
            )

            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )

            blocks.append(block)

        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

    def rebuild_t2s_transformer_from_state_dict(self, state_dict: dict, prefix: str = ""):
        runtime_dtype = self.ar_predict_layer.weight.dtype
        runtime_device = self.ar_predict_layer.weight.device

        def _tensor(name: str) -> torch.Tensor:
            return state_dict[name].to(device=runtime_device, dtype=runtime_dtype)

        blocks = []
        for i in range(self.num_layers):
            layer_prefix = f"{prefix}h.layers.{i}."
            t2smlp = T2SMLP(
                _tensor(layer_prefix + "linear1.weight"),
                _tensor(layer_prefix + "linear1.bias"),
                _tensor(layer_prefix + "linear2.weight"),
                _tensor(layer_prefix + "linear2.bias"),
            )
            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                _tensor(layer_prefix + "self_attn.in_proj_weight"),
                _tensor(layer_prefix + "self_attn.in_proj_bias"),
                _tensor(layer_prefix + "self_attn.out_proj.weight"),
                _tensor(layer_prefix + "self_attn.out_proj.bias"),
                _tensor(layer_prefix + "norm1.weight"),
                _tensor(layer_prefix + "norm1.bias"),
                self.h.layers[i].norm1.eps if self.h is not None else 1e-5,
                _tensor(layer_prefix + "norm2.weight"),
                _tensor(layer_prefix + "norm2.bias"),
                self.h.layers[i].norm2.eps if self.h is not None else 1e-5,
            )
            blocks.append(block)
        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

    def release_inference_only_unused_modules(self):
        # In inference runtime we only use infer_panel* paths backed by t2s_transformer.
        self.h = None
        self.loss_fct = None
        self.ar_accuracy_metric = None

    def _require_h_transformer(self):
        if self.h is None:
            raise RuntimeError("self.h has been released in inference-only runtime; use infer_panel* paths instead")

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask_left(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)

        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        # x_attn_mask[:, x_len]=False
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )

        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)

        return xy_pos, xy_attn_mask, targets

    def forward(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """
        self._require_h_transformer()

        reject_y, reject_y_lens = make_reject_y(y, y_lens)

        xy_pos, xy_attn_mask, targets = self.make_input_data(x, x_lens, y, y_lens, bert_feature)

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        x_len = x_lens.max()
        logits = self.ar_predict_layer(xy_dec[:, x_len-1:])

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(
            x, x_lens, reject_y, reject_y_lens, bert_feature
        )

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )
        x_len = x_lens.max()
        reject_logits = self.ar_predict_layer(reject_xy_dec[:, x_len-1:])

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        loss_1 = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.permute(0, 2, 1).detach(), targets).item()

        A_logits, R_logits = get_batch_logps(logits, reject_logits, targets, reject_targets)
        loss_2, _, _ = dpo_loss(A_logits, R_logits, 0, 0, 0.2, reference_free=True)

        loss = loss_1 + loss_2

        return loss, acc

    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """
        self._require_h_transformer()
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask_left(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len-1:]).permute(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        loss = F.cross_entropy(logits, targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(
        self,
        x,
        x_lens,
        prompts,
        bert_feature,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        self._require_h_transformer()
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        for _ in range(1500):
            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)
            # x 和逐渐增长的 y 一起输入给模型
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(y.device)

            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                break
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            # import os
            # os._exit(2333)
            y = torch.concat([y, samples], dim=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(y_mask_int, (0, 1), value=1)
        # 错位
        return targets[:, :-1], targets

    def infer_panel_batch_infer(
        self,
        x: List[torch.LongTensor],  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        if prompts is None:
            print("Warning: Prompt free is not supported batch_infer! switch to naive_infer")
            return self.infer_panel_naive_batched(
                x,
                x_lens,
                prompts,
                bert_feature,
                top_k=top_k,
                top_p=top_p,
                early_stop_num=early_stop_num,
                temperature=temperature,
                **kwargs,
            )

        max_len = kwargs.get("max_len", x_lens.max())
        probe = _get_benchmark_probe(self)
        torch_sdpa = bool(kwargs.get("torch_sdpa", True))
        disable_batch_shrink = bool(kwargs.get("disable_batch_shrink", False))
        batch_shrink_when_active_lte = int(kwargs.get("batch_shrink_when_active_lte", 0))
        stable_batch_remap = bool(kwargs.get("stable_batch_remap", False))
        x_list = []
        t_probe = time.perf_counter() if probe is not None else 0.0
        for x_item, bert_item in zip(x, bert_feature):
            # max_len = max(max_len, x_item.shape[0], bert_item.shape[1])
            x_item = self.ar_text_embedding(x_item.unsqueeze(0))
            x_item = x_item + self.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position(x_item).squeeze(0)
            # x_item = F.pad(x_item,(0,0,0,max_len-x_item.shape[0]),value=0) if x_item.shape[0]<max_len else x_item  ### padding right
            x_item = (
                F.pad(x_item, (0, 0, max_len - x_item.shape[0], 0), value=0) if x_item.shape[0] < max_len else x_item
            )  ### padding left
            x_list.append(x_item)
        if probe is not None:
            _probe_add(probe, "text_embed_sec", time.perf_counter() - t_probe)
            _probe_inc(probe, "batch_calls")
            _probe_inc(probe, "batch_items_total", len(x_list))
        x: torch.Tensor = torch.stack(x_list, dim=0)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        stop = False

        k_cache = None
        v_cache = None
        cache_len = 0
        ###################  first step ##########################
        assert y is not None, "Error: Prompt free is not supported batch_infer!"
        ref_free = False

        t_probe = time.perf_counter() if probe is not None else 0.0
        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_lens = torch.LongTensor([y_emb.shape[1]] * y_emb.shape[0]).to(x.device)
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        ##### create mask #####
        bsz = x.shape[0]
        src_len = x_len + y_len
        y_paddind_mask = make_pad_mask_left(y_lens, y_len)
        x_paddind_mask = make_pad_mask_left(x_lens, max_len)

        # (bsz, x_len + y_len)
        padding_mask = torch.concat([x_paddind_mask, y_paddind_mask], dim=1)

        x_mask = F.pad(
            torch.zeros(x_len, x_len, dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )

        y_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1),
            (x_len, 0),
            value=False,
        )

        causal_mask = torch.concat([x_mask, y_mask], dim=0).view(1, src_len, src_len).repeat(bsz, 1, 1).to(x.device)
        # padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2) ### [b, x+y, x+y]
        ### 上面是错误的，会导致padding的token被"看见"

        # 正确的padding_mask应该是：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6]]

        padding_mask = padding_mask.view(bsz, 1, src_len).repeat(1, src_len, 1)

        prompt_attn_mask: torch.Tensor = causal_mask.logical_or(padding_mask)
        prompt_attn_mask = prompt_attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1).bool()
        if probe is not None:
            _probe_add(probe, "prompt_prep_sec", time.perf_counter() - t_probe)

        # 正确的attn_mask应该是这样的：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5,   6]]

        ###### decode #####
        y_list = [None] * y.shape[0]
        batch_idx_map = list(range(y.shape[0]))
        idx_list = [None] * y.shape[0]
        active_rows = torch.ones((y.shape[0],), dtype=torch.bool, device=x.device)
        y_buffer = _alloc_token_buffer(y, MAX_AR_DECODE_STEPS)
        curr_y_len = prefix_len
        decode_attn_mask_full = torch.zeros(
            (bsz, self.num_head, 1, src_len + MAX_AR_DECODE_STEPS),
            dtype=torch.bool,
            device=x.device,
        )
        _probe_update_rss_peak(probe, -1)
        decode_attn_mask_full[:, :, :, :src_len] = prompt_attn_mask[:, :, -1:, :]
        decode_attn_mask = None
        for idx in range(MAX_AR_DECODE_STEPS):
            _probe_inc(probe, "decode_steps")
            if idx == 0:
                t_probe = time.perf_counter() if probe is not None else 0.0
                xy_dec, k_cache, v_cache, cache_len = self.t2s_transformer.process_prompt(
                    xy_pos,
                    prompt_attn_mask,
                    MAX_AR_DECODE_STEPS,
                    None,
                    torch_sdpa,
                )
                if probe is not None:
                    _probe_add(probe, "decode_transformer_sec", time.perf_counter() - t_probe)
                    _probe_inc(probe, "process_prompt_calls")
                    _probe_update_rss_peak(probe, idx)
            else:
                t_probe = time.perf_counter() if probe is not None else 0.0
                xy_dec, k_cache, v_cache, cache_len = self.t2s_transformer.decode_next_token(
                    xy_pos,
                    k_cache,
                    v_cache,
                    cache_len,
                    decode_attn_mask,
                    torch_sdpa,
                )
                if probe is not None:
                    _probe_add(probe, "decode_transformer_sec", time.perf_counter() - t_probe)
                    _probe_inc(probe, "decode_next_token_calls")
                    _probe_update_rss_peak(probe, idx)
            t_probe = time.perf_counter() if probe is not None else 0.0
            logits = self.ar_predict_layer(xy_dec[:, -1])
            if probe is not None:
                _probe_add(probe, "logits_sec", time.perf_counter() - t_probe)
                _probe_update_rss_peak(probe, idx)

            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1] 

            y = y_buffer[:, :curr_y_len]
            if probe is not None:
                _probe_add(probe, "sample_history_tokens_sum", curr_y_len)
                _probe_add(probe, "sample_batch_items_sum", y.shape[0])
            sample_logits = logits
            sample_y = y
            restore_order = None
            if stable_batch_remap and len(batch_idx_map) > 1:
                logical_order_list = [idx_ for idx_, _ in sorted(enumerate(batch_idx_map), key=lambda item: item[1])]
                if logical_order_list != list(range(len(batch_idx_map))):
                    logical_order = torch.tensor(logical_order_list, device=logits.device, dtype=torch.long)
                    sample_logits = torch.index_select(logits, dim=0, index=logical_order)
                    sample_y = torch.index_select(y, dim=0, index=logical_order)
                    restore_order = torch.empty_like(logical_order)
                    restore_order[logical_order] = torch.arange(logical_order.numel(), device=logits.device)
                    if probe is not None:
                        _probe_inc(probe, "sample_reorder_events")
            t_probe = time.perf_counter() if probe is not None else 0.0
            samples = sample(
                sample_logits,
                sample_y,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
            )[0]
            tokens = torch.argmax(sample_logits, dim=-1)
            if restore_order is not None:
                samples = torch.index_select(samples, dim=0, index=restore_order)
                tokens = torch.index_select(tokens, dim=0, index=restore_order)
            if probe is not None:
                _probe_add(probe, "sample_sec", time.perf_counter() - t_probe)
            y_buffer[:, curr_y_len : curr_y_len + 1] = samples
            curr_y_len += 1

            ####### 移除batch中已经生成完毕的序列,进一步优化计算量
            t_probe = time.perf_counter() if probe is not None else 0.0
            reserved_idx_of_batch_for_y = None
            should_shrink = False
            if (self.EOS in samples[:, 0]) or (self.EOS in tokens):  ###如果生成到EOS，则停止
                active_count_before_remove = int(active_rows.sum().item())
                l1 = samples[:, 0] == self.EOS
                l2 = tokens == self.EOS
                l = active_rows.logical_and(l1.logical_or(l2))
                removed_idx_of_batch_for_y = torch.where(l == True)[0].tolist()
                reserved_idx_of_batch_for_y = torch.where(active_rows.logical_and(l == False))[0]
                # batch_indexs = torch.tensor(batch_idx_map, device=y.device)[removed_idx_of_batch_for_y]
                for i in removed_idx_of_batch_for_y:
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y_buffer[i, : curr_y_len - 1].clone()

                active_count_after_remove = int(reserved_idx_of_batch_for_y.numel())
                should_shrink = (
                    (not disable_batch_shrink)
                    and (
                        batch_shrink_when_active_lte <= 0
                        or active_count_after_remove <= batch_shrink_when_active_lte
                    )
                )
                if probe is not None:
                    reserved_idx_list = reserved_idx_of_batch_for_y.tolist()
                    prefix_compactable = reserved_idx_list == list(range(len(reserved_idx_list)))
                    suffix_removed = removed_idx_of_batch_for_y == list(
                        range(active_count_after_remove, active_count_before_remove)
                    )
                    _probe_append(
                        probe,
                        "shrink_events_trace",
                        {
                            "decode_idx": int(idx),
                            "active_count_before_remove": active_count_before_remove,
                            "active_count_after_remove": active_count_after_remove,
                            "removed_idx": removed_idx_of_batch_for_y,
                            "reserved_idx": reserved_idx_list,
                            "should_shrink": bool(should_shrink),
                            "prefix_compactable": bool(prefix_compactable),
                            "suffix_removed": bool(suffix_removed),
                        },
                    )

                if not should_shrink:
                    if removed_idx_of_batch_for_y:
                        active_rows[torch.tensor(removed_idx_of_batch_for_y, device=active_rows.device)] = False
                    if probe is not None and not disable_batch_shrink:
                        _probe_inc(probe, "batch_shrink_deferred_events")
                        _probe_inc(probe, "batch_shrink_deferred_items", int(len(removed_idx_of_batch_for_y)))
                else:
                    if not stable_batch_remap:
                        batch_idx_map = [batch_idx_map[i] for i in reserved_idx_of_batch_for_y.tolist()]
            if probe is not None:
                _probe_add(probe, "stop_check_sec", time.perf_counter() - t_probe)

            # 只保留batch中未生成完毕的序列
            if reserved_idx_of_batch_for_y is not None and should_shrink:
                t_probe = time.perf_counter() if probe is not None else 0.0
                removed_count = int(len(removed_idx_of_batch_for_y))
                if stable_batch_remap:
                    active_count_after_remove = int(reserved_idx_of_batch_for_y.numel())
                    removed_set = set(removed_idx_of_batch_for_y)
                    hole_indices = [row for row in removed_idx_of_batch_for_y if row < active_count_after_remove]
                    donor = active_count_before_remove - 1
                    moved_rows = 0
                    for hole in hole_indices:
                        while donor in removed_set:
                            donor -= 1
                        if donor < active_count_after_remove:
                            break
                        _move_token_buffer_row(y_buffer, hole, donor, curr_y_len)
                        _move_mask_row(decode_attn_mask_full, hole, donor)
                        samples[hole] = samples[donor]
                        if k_cache is not None:
                            for i in range(len(k_cache)):
                                _move_cache_buffer_row(k_cache[i], hole, donor, cache_len)
                                _move_cache_buffer_row(v_cache[i], hole, donor, cache_len)
                        batch_idx_map[hole] = batch_idx_map[donor]
                        donor -= 1
                        moved_rows += 1
                    y_buffer = y_buffer[:active_count_after_remove]
                    decode_attn_mask_full = decode_attn_mask_full[:active_count_after_remove]
                    samples = samples[:active_count_after_remove]
                    batch_idx_map = batch_idx_map[:active_count_after_remove]
                    if k_cache is not None:
                        for i in range(len(k_cache)):
                            k_cache[i] = k_cache[i][:active_count_after_remove]
                            v_cache[i] = v_cache[i][:active_count_after_remove]
                    if probe is not None:
                        _probe_inc(probe, "batch_remap_events")
                        _probe_inc(probe, "batch_remap_moved_rows", moved_rows)
                else:
                    y_buffer = _compact_token_buffer(y_buffer, reserved_idx_of_batch_for_y, curr_y_len)
                    decode_attn_mask_full = _compact_decode_attn_mask_full(
                        decode_attn_mask_full, reserved_idx_of_batch_for_y, src_len
                    )
                    samples = torch.index_select(samples, dim=0, index=reserved_idx_of_batch_for_y)
                    if k_cache is not None:
                        for i in range(len(k_cache)):
                            k_cache[i] = _compact_cache_buffer(k_cache[i], reserved_idx_of_batch_for_y, cache_len)
                            v_cache[i] = _compact_cache_buffer(v_cache[i], reserved_idx_of_batch_for_y, cache_len)
                active_rows = torch.ones((y_buffer.shape[0],), dtype=torch.bool, device=active_rows.device)
                if probe is not None:
                    _probe_add(probe, "shrink_sec", time.perf_counter() - t_probe)
                    _probe_inc(probe, "batch_shrink_events")
                    _probe_inc(probe, "batch_shrink_items", removed_count)
                    _probe_update_rss_peak(probe, idx)
            elif reserved_idx_of_batch_for_y is not None and disable_batch_shrink and probe is not None:
                _probe_inc(probe, "batch_shrink_skipped_events")
                _probe_inc(probe, "batch_shrink_items", int(len(removed_idx_of_batch_for_y)))

            if (early_stop_num != -1 and (curr_y_len - prefix_len) > early_stop_num) or idx == MAX_AR_DECODE_STEPS - 1:
                stop = True
                for i, batch_index in enumerate(batch_idx_map):
                    batch_index = batch_idx_map[i]
                    if idx_list[batch_index] is None:
                        idx_list[batch_index] = idx
                        y_list[batch_index] = y_buffer[i, : curr_y_len - 1].clone()

            if None not in idx_list:
                stop = True

            if stop:
                if curr_y_len == 0:
                    y_buffer[:, 0:1] = torch.zeros_like(samples)
                    curr_y_len = 1
                break

            decode_attn_mask = decode_attn_mask_full[:, :, :, : cache_len + 1]

            ####################### update next step ###################################
            t_probe = time.perf_counter() if probe is not None else 0.0
            y_emb = self.ar_audio_embedding(samples)
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)
            if probe is not None:
                _probe_add(probe, "next_pos_sec", time.perf_counter() - t_probe)
                _probe_update_rss_peak(probe, idx)

        if None in idx_list:
            for i in range(x.shape[0]):
                if idx_list[i] is None:
                    idx_list[i] = MAX_AR_DECODE_STEPS - 1  ###如果没有生成到EOS，就用最大长度代替

        if probe is not None:
            _probe_add(probe, "generated_tokens_sum", max(curr_y_len - prefix_len, 0))
            _probe_update_rss_peak(probe, idx if curr_y_len > 0 else -1)
        if ref_free:
            return y_list, [0] * x.shape[0]
        # print(idx_list)
        return y_list, idx_list

    def infer_panel_naive_batched(
        self,
        x: List[torch.LongTensor],  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        y_list = []
        idx_list = []
        for i in range(len(x)):
            y, idx = next(self.infer_panel_naive(
                x[i].unsqueeze(0),
                x_lens[i],
                prompts[i].unsqueeze(0) if prompts is not None else None,
                bert_feature[i].unsqueeze(0),
                top_k,
                top_p,
                early_stop_num,
                temperature,
                repetition_penalty,
                **kwargs,
            ))
            y_list.append(y[0])
            idx_list.append(idx)

        return y_list, idx_list

    def infer_panel_naive(
        self,
        x: torch.LongTensor,  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        streaming_mode: bool = False,
        chunk_length: int = 24,
        **kwargs,
    ):
        mute_emb_sim_matrix = kwargs.get("mute_emb_sim_matrix", None)
        chunk_split_thershold = kwargs.get("chunk_split_thershold", 0.3)
        check_token_num = 2
        probe = _get_benchmark_probe(self)
        torch_sdpa = bool(kwargs.get("torch_sdpa", True))


        t_probe = time.perf_counter() if probe is not None else 0.0
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        if probe is not None:
            _probe_add(probe, "text_embed_sec", time.perf_counter() - t_probe)
            _probe_inc(probe, "batch_calls")
            _probe_inc(probe, "batch_items_total", int(x.shape[0]))

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        # print(1111111,self.num_layers)

        k_cache = None
        v_cache = None
        cache_len = 0
        ###################  first step ##########################
        t_probe = time.perf_counter() if probe is not None else 0.0
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_attn_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = (
            torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )
        if probe is not None:
            _probe_add(probe, "prompt_prep_sec", time.perf_counter() - t_probe)

        token_counter = 0
        curr_ptr = prefix_len
        y_buffer = _alloc_token_buffer(y, MAX_AR_DECODE_STEPS)
        curr_y_len = prefix_len
        for idx in range(MAX_AR_DECODE_STEPS):
            token_counter+=1
            _probe_inc(probe, "decode_steps")
            if xy_attn_mask is not None:
                t_probe = time.perf_counter() if probe is not None else 0.0
                xy_dec, k_cache, v_cache, cache_len = self.t2s_transformer.process_prompt(
                    xy_pos,
                    xy_attn_mask,
                    MAX_AR_DECODE_STEPS,
                    None,
                    torch_sdpa,
                )
                if probe is not None:
                    _probe_add(probe, "decode_transformer_sec", time.perf_counter() - t_probe)
                    _probe_inc(probe, "process_prompt_calls")
            else:
                t_probe = time.perf_counter() if probe is not None else 0.0
                xy_dec, k_cache, v_cache, cache_len = self.t2s_transformer.decode_next_token(
                    xy_pos,
                    k_cache,
                    v_cache,
                    cache_len,
                    None,
                    torch_sdpa,
                )
                if probe is not None:
                    _probe_add(probe, "decode_transformer_sec", time.perf_counter() - t_probe)
                    _probe_inc(probe, "decode_next_token_calls")

            t_probe = time.perf_counter() if probe is not None else 0.0
            logits = self.ar_predict_layer(xy_dec[:, -1])
            if probe is not None:
                _probe_add(probe, "logits_sec", time.perf_counter() - t_probe)

            if idx == 0:
                xy_attn_mask = None
            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]

            y = y_buffer[:, :curr_y_len]
            if probe is not None:
                _probe_add(probe, "sample_history_tokens_sum", curr_y_len)
                _probe_add(probe, "sample_batch_items_sum", y.shape[0])
            t_probe = time.perf_counter() if probe is not None else 0.0
            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]
            if probe is not None:
                _probe_add(probe, "sample_sec", time.perf_counter() - t_probe)
            y_buffer[:, curr_y_len : curr_y_len + 1] = samples
            curr_y_len += 1

            t_probe = time.perf_counter() if probe is not None else 0.0
            if early_stop_num != -1 and (curr_y_len - prefix_len) > early_stop_num:
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
                curr_y_len -= 1
                token_counter -= 1

            if idx == MAX_AR_DECODE_STEPS - 1:
                stop = True
            if probe is not None:
                _probe_add(probe, "stop_check_sec", time.perf_counter() - t_probe)

            if stop:
                if curr_y_len == 0:
                    y_buffer[:, 0:1] = torch.zeros_like(samples)
                    curr_y_len = 1
                # print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                if streaming_mode:
                    final_y = y_buffer[:, :curr_y_len]
                    yield final_y[:, curr_ptr:] if curr_ptr < final_y.shape[1] else None, True
                break


            if streaming_mode and (mute_emb_sim_matrix is not None) and (token_counter >= chunk_length+check_token_num):
                active_y = y_buffer[:, :curr_y_len]
                score = mute_emb_sim_matrix[active_y[0, curr_ptr:]] - chunk_split_thershold
                score[score<0]=-1
                score[:-1]=score[:-1]+score[1:] ##考虑连续两个token
                argmax_idx = score.argmax()

                if score[argmax_idx]>=0 and argmax_idx+1>=chunk_length: 
                    print(f"\n\ncurr_ptr:{curr_ptr}")
                    yield active_y[:, curr_ptr:], False
                    token_counter -= argmax_idx+1
                    curr_ptr += argmax_idx+1


            elif streaming_mode and (mute_emb_sim_matrix is None) and (token_counter >= chunk_length):
                active_y = y_buffer[:, :curr_y_len]
                yield active_y[:, -token_counter:], False
                curr_ptr+=token_counter
                token_counter = 0
                


            ####################### update next step ###################################
            t_probe = time.perf_counter() if probe is not None else 0.0
            y_emb = self.ar_audio_embedding(samples)
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)
            if probe is not None:
                _probe_add(probe, "next_pos_sec", time.perf_counter() - t_probe)



        if not streaming_mode:
            y = y_buffer[:, :curr_y_len]
            if probe is not None:
                _probe_add(probe, "generated_tokens_sum", max(curr_y_len - prefix_len, 0))
            if ref_free:
                yield y, 0
            yield y, idx



    def infer_panel(
        self,
        x: torch.LongTensor,  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        return next(self.infer_panel_naive(
            x, x_lens, prompts, bert_feature, top_k, top_p, early_stop_num, temperature, repetition_penalty, **kwargs
        ))
