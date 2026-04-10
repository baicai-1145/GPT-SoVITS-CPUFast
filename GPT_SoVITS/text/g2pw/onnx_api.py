# This code is modified from https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw
# This code is modified from https://github.com/GitYCC/g2pW

import json
import os
import pickle
import warnings
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime
from opencc import OpenCC
from pypinyin import Style, pinyin
from tokenizers import Tokenizer

from ..zh_normalization.char_convert import tranditional_to_simplified
from .dataset import get_char_phoneme_labels, get_phoneme_labels, prepare_onnx_input
from .utils import load_config

onnxruntime.set_default_logger_severity(3)
try:
    onnxruntime.preload_dlls()
except Exception:
    pass
warnings.filterwarnings("ignore")

model_version = "1.1"
STATIC_ASSETS_CACHE_VERSION = 1


class _TokenizerAdapter:
    def __init__(self, tokenizer_file: str):
        self._tokenizer = Tokenizer.from_file(tokenizer_file)
        self._unk_id = self._tokenizer.token_to_id("[UNK]")

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer.encode(text, add_special_tokens=False).tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for token in tokens:
            token_id = self._tokenizer.token_to_id(token)
            ids.append(self._unk_id if token_id is None else token_id)
        return ids


def predict(session, onnx_input: Dict[str, Any], labels: List[str]) -> Tuple[List[str], List[float]]:
    all_preds = []
    all_confidences = []
    probs = session.run(
        [],
        {
            "input_ids": onnx_input["input_ids"],
            "token_type_ids": onnx_input["token_type_ids"],
            "attention_mask": onnx_input["attention_masks"],
            "phoneme_mask": onnx_input["phoneme_masks"],
            "char_ids": onnx_input["char_ids"],
            "position_ids": onnx_input["position_ids"],
        },
    )[0]

    preds = np.argmax(probs, axis=1).tolist()
    max_probs = []
    for index, arr in zip(preds, probs.tolist()):
        max_probs.append(arr[index])
    all_preds += [labels[pred] for pred in preds]
    all_confidences += max_probs

    return all_preds, all_confidences


def _load_json_from_candidates(filename: str, candidate_dirs: List[str]) -> Dict[str, Any]:
    for candidate_dir in candidate_dirs:
        if not candidate_dir:
            continue
        json_path = os.path.join(candidate_dir, filename)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as fr:
                return json.load(fr)
    raise FileNotFoundError(f"Cannot locate {filename} in candidate dirs: {candidate_dirs}")


def _find_first_existing_file(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Files not found: {paths}")


def _get_static_asset_sources(model_dir: str) -> Dict[str, str]:
    default_asset_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "G2PWModel"))
    candidate_asset_dirs = [model_dir, default_asset_dir]
    return {
        "polyphonic_chars": os.path.join(model_dir, "POLYPHONIC_CHARS.txt"),
        "monophonic_chars": os.path.join(model_dir, "MONOPHONIC_CHARS.txt"),
        "bopomofo_to_pinyin": _find_first_existing_file(
            os.path.join(candidate_asset_dirs[0], "bopomofo_to_pinyin_wo_tune_dict.json"),
            os.path.join(candidate_asset_dirs[1], "bopomofo_to_pinyin_wo_tune_dict.json"),
        ),
        "char_bopomofo": _find_first_existing_file(
            os.path.join(candidate_asset_dirs[0], "char_bopomofo_dict.json"),
            os.path.join(candidate_asset_dirs[1], "char_bopomofo_dict.json"),
        ),
    }


def _get_static_assets_cache_path(model_dir: str, use_char_phoneme: bool, use_mask: bool) -> str:
    return os.path.join(
        model_dir,
        f"g2pw_static_assets_v{STATIC_ASSETS_CACHE_VERSION}_cp{int(use_char_phoneme)}_mask{int(use_mask)}.pickle",
    )


def _build_static_assets(model_dir: str, use_char_phoneme: bool, use_mask: bool) -> Dict[str, Any]:
    sources = _get_static_asset_sources(model_dir)
    polyphonic_chars = [
        line.split("\t") for line in open(sources["polyphonic_chars"], encoding="utf-8").read().strip().split("\n")
    ]
    monophonic_chars = [
        line.split("\t") for line in open(sources["monophonic_chars"], encoding="utf-8").read().strip().split("\n")
    ]
    labels, char2phonemes = (
        get_char_phoneme_labels(polyphonic_chars=polyphonic_chars)
        if use_char_phoneme
        else get_phoneme_labels(polyphonic_chars=polyphonic_chars)
    )
    chars = sorted(list(char2phonemes.keys()))
    char2id = {char: idx for idx, char in enumerate(chars)}
    char_phoneme_masks = (
        {
            char: [1 if i in char2phonemes[char] else 0 for i in range(len(labels))]
            for char in char2phonemes
        }
        if use_mask
        else None
    )
    non_polyphonic = {
        "一",
        "不",
        "和",
        "咋",
        "嗲",
        "剖",
        "差",
        "攢",
        "倒",
        "難",
        "奔",
        "勁",
        "拗",
        "肖",
        "瘙",
        "誒",
        "泊",
        "听",
        "噢",
    }
    non_monophonic = {"似", "攢"}
    polyphonic_chars_new = set(chars)
    for char in non_polyphonic:
        polyphonic_chars_new.discard(char)
    monophonic_chars_dict = {char: phoneme for char, phoneme in monophonic_chars}
    for char in non_monophonic:
        monophonic_chars_dict.pop(char, None)

    with open(sources["bopomofo_to_pinyin"], "r", encoding="utf-8") as fr:
        bopomofo_convert_dict = json.load(fr)
    with open(sources["char_bopomofo"], "r", encoding="utf-8") as fr:
        char_bopomofo_dict = json.load(fr)

    return {
        "cache_version": STATIC_ASSETS_CACHE_VERSION,
        "source_mtimes": {key: os.path.getmtime(path) for key, path in sources.items()},
        "labels": labels,
        "char2phonemes": char2phonemes,
        "chars": chars,
        "char2id": char2id,
        "char_phoneme_masks": char_phoneme_masks,
        "polyphonic_chars_new": polyphonic_chars_new,
        "monophonic_chars_dict": monophonic_chars_dict,
        "bopomofo_convert_dict": bopomofo_convert_dict,
        "char_bopomofo_dict": char_bopomofo_dict,
    }


def _load_or_build_static_assets(model_dir: str, use_char_phoneme: bool, use_mask: bool) -> Dict[str, Any]:
    cache_path = _get_static_assets_cache_path(model_dir, use_char_phoneme=use_char_phoneme, use_mask=use_mask)
    sources = _get_static_asset_sources(model_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fr:
                cached = pickle.load(fr)
            if cached.get("cache_version") == STATIC_ASSETS_CACHE_VERSION and all(
                cached["source_mtimes"].get(key) == os.path.getmtime(path) for key, path in sources.items()
            ):
                return cached
        except Exception:
            pass

    built = _build_static_assets(model_dir, use_char_phoneme=use_char_phoneme, use_mask=use_mask)
    try:
        with open(cache_path, "wb") as fw:
            pickle.dump(built, fw, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return built


def download_and_decompress(model_dir: str = "G2PWModel/"):
    if not os.path.exists(model_dir):
        import requests

        parent_directory = os.path.dirname(model_dir)
        zip_dir = os.path.join(parent_directory, "G2PWModel_1.1.zip")
        extract_dir = os.path.join(parent_directory, "G2PWModel_1.1")
        extract_dir_new = os.path.join(parent_directory, "G2PWModel")
        print("Downloading g2pw model...")
        modelscope_url = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"
        with requests.get(modelscope_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_dir, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Extracting g2pw model...")
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(parent_directory)

        os.rename(extract_dir, extract_dir_new)

    return model_dir


class _G2PWBaseOnnxConverter:
    def __init__(
        self,
        model_dir: str = "G2PWModel/",
        style: str = "bopomofo",
        model_source: str = None,
        enable_non_tradional_chinese: bool = False,
    ):
        self.model_dir = download_and_decompress(model_dir)
        self.config = load_config(config_path=os.path.join(self.model_dir, "config.py"), use_default=True)

        self.model_source = model_source if model_source else self.config.model_source
        self.enable_opencc = enable_non_tradional_chinese
        tokenizer_file = os.path.join(self.model_source, "tokenizer.json")
        self.tokenizer = _TokenizerAdapter(tokenizer_file=tokenizer_file)
        static_assets = _load_or_build_static_assets(
            self.model_dir,
            use_char_phoneme=self.config.use_char_phoneme,
            use_mask=self.config.use_mask,
        )
        self.labels = static_assets["labels"]
        self.char2phonemes = static_assets["char2phonemes"]
        self.chars = static_assets["chars"]
        self.char2id = static_assets["char2id"]
        self.char_phoneme_masks = static_assets["char_phoneme_masks"]
        self.polyphonic_chars_new = static_assets["polyphonic_chars_new"]
        self.monophonic_chars_dict = static_assets["monophonic_chars_dict"]
        self.bopomofo_convert_dict = static_assets["bopomofo_convert_dict"]
        self.char_bopomofo_dict = static_assets["char_bopomofo_dict"]

        self.style_convert_func = {
            "bopomofo": lambda x: x,
            "pinyin": self._convert_bopomofo_to_pinyin,
        }[style]

        if self.enable_opencc:
            self.cc = OpenCC("s2tw")
        self.enable_sentence_dedup = os.getenv("g2pw_sentence_dedup", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        # 聚焦到多音字附近上下文，默认左右各16字；设为0表示关闭裁剪（整句）。
        self.polyphonic_context_chars = max(0, int(os.getenv("g2pw_polyphonic_context_chars", "16")))

    def _convert_bopomofo_to_pinyin(self, bopomofo: str) -> str:
        tone = bopomofo[-1]
        assert tone in "12345"
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        print(f'Warning: "{bopomofo}" cannot convert to pinyin')
        return None

    def __call__(self, sentences: List[str]) -> List[List[str]]:
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.enable_opencc:
            translated_sentences = []
            for sent in sentences:
                translated_sent = self.cc.convert(sent)
                assert len(translated_sent) == len(sent)
                translated_sentences.append(translated_sent)
            sentences = translated_sentences

        texts, model_query_ids, result_query_ids, sent_ids, partial_results = self._prepare_data(sentences=sentences)
        if len(texts) == 0:
            return partial_results

        model_input = prepare_onnx_input(
            tokenizer=self.tokenizer,
            labels=self.labels,
            char2phonemes=self.char2phonemes,
            chars=self.chars,
            texts=texts,
            query_ids=model_query_ids,
            use_mask=self.config.use_mask,
            window_size=None,
            char2id=self.char2id,
            char_phoneme_masks=self.char_phoneme_masks,
        )

        if not model_input:
            return partial_results

        if self.enable_sentence_dedup:
            preds, _confidences = self._predict_with_sentence_dedup(model_input=model_input, texts=texts)
        else:
            preds, _confidences = self._predict(model_input=model_input)

        if self.config.use_char_phoneme:
            preds = [pred.split(" ")[1] for pred in preds]

        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, result_query_ids, preds):
            results[sent_id][query_id] = self.style_convert_func(pred)

        return results

    def _prepare_data(
        self, sentences: List[str]
    ) -> Tuple[List[str], List[int], List[int], List[int], List[List[str]]]:
        texts, model_query_ids, result_query_ids, sent_ids, partial_results = [], [], [], [], []
        for sent_id, sent in enumerate(sentences):
            sent_s = tranditional_to_simplified(sent)
            pypinyin_result = pinyin(sent_s, neutral_tone_with_five=True, style=Style.TONE3)
            partial_result = [None] * len(sent)
            polyphonic_indices: List[int] = []
            for i, char in enumerate(sent):
                if char in self.polyphonic_chars_new:
                    polyphonic_indices.append(i)
                elif char in self.monophonic_chars_dict:
                    partial_result[i] = self.style_convert_func(self.monophonic_chars_dict[char])
                elif char in self.char_bopomofo_dict:
                    partial_result[i] = pypinyin_result[i][0]
                else:
                    partial_result[i] = pypinyin_result[i][0]

            if polyphonic_indices:
                if self.polyphonic_context_chars > 0:
                    left = max(0, polyphonic_indices[0] - self.polyphonic_context_chars)
                    right = min(len(sent), polyphonic_indices[-1] + self.polyphonic_context_chars + 1)
                    sent_for_predict = sent[left:right]
                    query_offset = left
                else:
                    sent_for_predict = sent
                    query_offset = 0

                for index in polyphonic_indices:
                    texts.append(sent_for_predict)
                    model_query_ids.append(index - query_offset)
                    result_query_ids.append(index)
                    sent_ids.append(sent_id)

            partial_results.append(partial_result)
        return texts, model_query_ids, result_query_ids, sent_ids, partial_results

    def _predict(self, model_input: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        raise NotImplementedError

    def _predict_with_sentence_dedup(
        self, model_input: Dict[str, Any], texts: List[str]
    ) -> Tuple[List[str], List[float]]:
        if len(texts) <= 1:
            return self._predict(model_input=model_input)

        grouped_indices: Dict[str, List[int]] = {}
        for idx, text in enumerate(texts):
            grouped_indices.setdefault(text, []).append(idx)

        if all(len(indices) == 1 for indices in grouped_indices.values()):
            return self._predict(model_input=model_input)

        preds: List[str] = [""] * len(texts)
        confidences: List[float] = [0.0] * len(texts)
        for indices in grouped_indices.values():
            group_input = {name: value[indices] for name, value in model_input.items()}
            if len(indices) > 1:
                for name in ("input_ids", "token_type_ids", "attention_masks"):
                    group_input[name] = group_input[name][:1]

            group_preds, group_confidences = self._predict(model_input=group_input)
            for output_idx, pred, confidence in zip(indices, group_preds, group_confidences):
                preds[output_idx] = pred
                confidences[output_idx] = confidence

        return preds, confidences


class G2PWOnnxConverter(_G2PWBaseOnnxConverter):
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

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2

        onnx_path = _find_first_existing_file(
            os.path.join(self.model_dir, "g2pW.onnx"),
            os.path.join(self.model_dir, "g2pw.onnx"),
        )

        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            self.session_g2pw = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            self.session_g2pw = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

    def _predict(self, model_input: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        return predict(session=self.session_g2pw, onnx_input=model_input, labels=self.labels)
