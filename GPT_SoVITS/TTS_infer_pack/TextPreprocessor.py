import os
import sys
import threading
from typing import Dict, List, Tuple

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
from text.LangSegmenter import LangSegmenter
from text import chinese
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method

from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
punctuation = set(["!", "?", "…", ",", ".", "-"])


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts: str, threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


class TextPreprocessor:
    def __init__(self, bert_model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, device: torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_lock = threading.RLock()
        self.non_zh_bert_cache = {}

    def _should_log(self) -> bool:
        mode = os.environ.get("GPTSOVITS_PREPROCESS_LOG", "auto").strip().lower()
        if mode in {"1", "true", "yes", "on"}:
            return True
        if mode in {"0", "false", "no", "off"}:
            return False
        return bool(getattr(sys.stdout, "isatty", lambda: False)())

    def _maybe_log(self, *args):
        if self._should_log():
            print(*args)

    def _iter_texts(self, texts):
        if self._should_log() and len(texts) > 1:
            return tqdm(texts)
        return texts

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        self._maybe_log(f"############ {i18n('切分文本')} ############")
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        if self._can_batch_zh_preprocess(texts, lang):
            return self._preprocess_batch_zh(texts, version)
        result = []
        self._maybe_log(f"############ {i18n('提取文本Bert特征')} ############")
        for text in self._iter_texts(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text == "":
                continue
            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def _can_batch_zh_preprocess(self, texts: List[str], lang: str) -> bool:
        if lang not in {"zh", "all_zh"}:
            return False
        if not texts:
            return False
        return all(re.search(r"[A-Za-z]", text) is None for text in texts)

    def _extract_pure_zh_text(self, text: str, version: str, final: bool = False):
        phones, word2ph, norm_text = self.clean_text_inf(text, "zh", version)
        if not final and len(phones) < 6:
            return self._extract_pure_zh_text("." + text, version, final=True)
        return phones, word2ph, norm_text

    def _preprocess_batch_zh(self, texts: List[str], version: str) -> List[Dict]:
        result = []
        prepared = []
        self._maybe_log(f"############ {i18n('提取文本Bert特征')} ############")
        for text in self._iter_texts(texts):
            phones, word2ph, norm_text = self._extract_pure_zh_text(text, version)
            if phones is None or norm_text == "":
                continue
            prepared.append((phones, word2ph, norm_text))

        if not prepared:
            return result

        with self.bert_lock:
            bert_list = self.get_bert_feature_batch(
                [item[2] for item in prepared],
                [item[1] for item in prepared],
            )

        for (phones, _word2ph, norm_text), bert_features in zip(prepared, bert_list):
            result.append(
                {
                    "phones": phones,
                    "bert_features": bert_features.to(self.device),
                    "norm_text": norm_text,
                }
            )
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
        self._maybe_log(i18n("实际输入的目标文本:"))
        self._maybe_log(text)

        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if not re.sub(r"\W+", "", text):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if text[-1] not in splits:
                text += "。" if lang != "en" else "."

            # 解决句子过长导致Bert报错的问题
            if len(text) > 510:
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        self._maybe_log(i18n("实际输入的目标文本(切句后):"))
        self._maybe_log(texts)
        return texts

    def segment_and_extract_feature_for_text(
        self, text: str, language: str, version: str = "v1"
    ) -> Tuple[list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language, version)

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False):
        with self.bert_lock:
            text = re.sub(r' {2,}', ' ', text)
            textlist = []
            langlist = []
            if language == "all_zh":
                for tmp in LangSegmenter.getTexts(text,"zh"):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "all_yue":
                for tmp in LangSegmenter.getTexts(text,"zh"):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "all_ja":
                for tmp in LangSegmenter.getTexts(text,"ja"):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "all_ko":
                for tmp in LangSegmenter.getTexts(text,"ko"):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "en":
                langlist.append("en")
                textlist.append(text)
            elif language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
                    if langlist:
                        if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                            textlist[-1] += tmp["text"]
                            continue
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            # print(textlist)
            # print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)

            if not final and len(phones) < 6:
                return self.get_phones_and_bert("." + text, language, version, final=True)

            return phones, bert, norm_text

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_bert_feature_batch(self, texts: List[str], word2ph_list: List[list]) -> List[torch.Tensor]:
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            hidden = torch.cat(res["hidden_states"][-3:-2], -1).cpu()

        feature_list = []
        for idx, (text, word2ph) in enumerate(zip(texts, word2ph_list)):
            assert len(word2ph) == len(text)
            char_feature = hidden[idx][1 : 1 + len(text)]
            repeat_counts = torch.tensor(word2ph, dtype=torch.long)
            phone_level_feature = torch.repeat_interleave(char_feature, repeat_counts, dim=0)
            feature_list.append(phone_level_feature.T)
        return feature_list

    def clean_text_inf(self, text: str, language: str, version: str = "v2"):
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            cache_key = len(phones)
            feature = self.non_zh_bert_cache.get(cache_key)
            if feature is None:
                feature = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.non_zh_bert_cache[cache_key] = feature

        return feature

    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result
