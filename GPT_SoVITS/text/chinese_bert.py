import json
import os
from types import SimpleNamespace
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


ACT2FN = {
    "gelu": _gelu,
}


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0),
            persistent=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        if self.training:
            attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(new_context_shape)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self_output = self.self(hidden_states, attention_mask)
        return self.output(self_output, hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def _get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - extended_attention_mask) * torch.finfo(dtype).min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attention_mask = self._get_extended_attention_mask(attention_mask, embedding_output.dtype)
        sequence_output, all_hidden_states = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        return {
            "last_hidden_state": sequence_output,
            "hidden_states": tuple(all_hidden_states) if output_hidden_states else None,
        }


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        return self.LayerNorm(hidden_states)


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.predictions(sequence_output)


class BertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = self.cls(outputs["last_hidden_state"])
        return {
            "logits": logits,
            "hidden_states": outputs["hidden_states"],
            "last_hidden_state": outputs["last_hidden_state"],
        }


class ChineseBertTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def _encode(self, text: str):
        return self.tokenizer.encode(text)

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        return_tensors: str = "pt",
        padding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        encodings = [self._encode(text) for text in texts]
        input_ids_list = [enc.ids for enc in encodings]
        token_type_ids_list = [enc.type_ids for enc in encodings]
        attention_mask_list = [enc.attention_mask for enc in encodings]

        max_length = max(len(ids) for ids in input_ids_list) if input_ids_list else 0
        if padding:
            pad_id = 0
            padded_input_ids = []
            padded_token_type_ids = []
            padded_attention_masks = []
            for input_ids, token_type_ids, attention_mask in zip(
                input_ids_list, token_type_ids_list, attention_mask_list
            ):
                pad_len = max_length - len(input_ids)
                padded_input_ids.append(input_ids + [pad_id] * pad_len)
                padded_token_type_ids.append(token_type_ids + [0] * pad_len)
                padded_attention_masks.append(attention_mask + [0] * pad_len)
            input_ids_list = padded_input_ids
            token_type_ids_list = padded_token_type_ids
            attention_mask_list = padded_attention_masks

        result = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        }
        if single_input:
            return {key: value for key, value in result.items()}
        return result


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_config(base_path: str):
    config_dict = _load_json(os.path.join(base_path, "config.json"))
    return SimpleNamespace(**config_dict)


def load_tokenizer(base_path: str) -> ChineseBertTokenizer:
    return ChineseBertTokenizer(os.path.join(base_path, "tokenizer.json"))


def load_model(base_path: str) -> BertForMaskedLM:
    config = _load_config(base_path)
    model = BertForMaskedLM(config)
    state_dict = torch.load(os.path.join(base_path, "pytorch_model.bin"), map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Failed to load Chinese BERT weights cleanly. missing={missing_keys}, unexpected={unexpected_keys}"
        )
    return model


def get_bert_feature(model, tokenizer, text: str, word2ph: List[int], device: Union[str, torch.device]) -> torch.Tensor:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs["hidden_states"][-3][0].cpu()[1:-1]

    assert len(word2ph) == len(text)
    phone_level_feature = []
    for idx, repeat_count in enumerate(word2ph):
        phone_level_feature.append(hidden[idx].repeat(repeat_count, 1))
    return torch.cat(phone_level_feature, dim=0).T


def get_bert_feature_batch(
    model,
    tokenizer,
    texts: List[str],
    word2ph_list: List[List[int]],
    device: Union[str, torch.device],
) -> List[torch.Tensor]:
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs["hidden_states"][-3].cpu()

    feature_list = []
    for idx, (text, word2ph) in enumerate(zip(texts, word2ph_list)):
        assert len(word2ph) == len(text)
        char_feature = hidden[idx][1 : 1 + len(text)]
        repeat_counts = torch.tensor(word2ph, dtype=torch.long)
        phone_level_feature = torch.repeat_interleave(char_feature, repeat_counts, dim=0)
        feature_list.append(phone_level_feature.T)
    return feature_list
