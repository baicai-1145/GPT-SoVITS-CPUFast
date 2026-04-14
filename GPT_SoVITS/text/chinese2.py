import os
import re
from functools import lru_cache

from pypinyin import Style
from pypinyin import pinyin
from pypinyin.constants import PHRASES_DICT
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
from text.zh_normalization.char_convert import tranditional_to_simplified
from text.zh_normalization.text_normlization import TextNormalizer

text_normalizer = TextNormalizer()

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba_fast
import logging

jieba_fast.setLogLevel(logging.CRITICAL)
from text import jieba_posseg_fast as psg
from text.phone_units import finalize_phone_units

# is_g2pw_str = os.environ.get("is_g2pw", "True")##默认开启
# is_g2pw = False#True if is_g2pw_str.lower() == 'true' else False
is_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False
if is_g2pw:
    # print("当前使用g2pw进行拼音推理")
    from text.g2pw.onnx_api import G2PWOnnxConverter
    from text.g2pw.pronunciation import correct_pronunciation, get_phrase_pronunciation

    parent_directory = os.path.dirname(current_file_path)
    g2pw = G2PWOnnxConverter(
        model_dir="GPT_SoVITS/text/G2PWModel",
        style="pinyin",
        model_source=os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        enable_non_tradional_chinese=True,
    )

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def g2p_with_phone_units(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph, phone_units = _g2p(sentences, return_phone_units=True)
    return phones, word2ph, phone_units


def _get_initials_finals(word):
    from pypinyin import lazy_pinyin

    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _word_has_g2pw_polyphonic_char(word: str) -> bool:
    return any(char in g2pw.polyphonic_chars_new for char in word)


@lru_cache(maxsize=8192)
def _get_phrase_level_pinyin_override(word: str):
    if not is_g2pw or len(word) <= 1:
        return None

    local_override = get_phrase_pronunciation(word)
    if local_override is not None and len(local_override) == len(word):
        return tuple(local_override)

    simplified_word = tranditional_to_simplified(word)
    if simplified_word != word:
        local_override = get_phrase_pronunciation(simplified_word)
        if local_override is not None and len(local_override) == len(word):
            return tuple(local_override)

    if not _word_has_g2pw_polyphonic_char(word):
        return None

    lookup_word = word if word in PHRASES_DICT else simplified_word if simplified_word in PHRASES_DICT else None
    if lookup_word is None:
        return None

    resolved = []
    for item in pinyin(lookup_word, neutral_tone_with_five=True, style=Style.TONE3):
        if not item or not item[0]:
            return None
        resolved.append(item[0])

    if len(resolved) != len(word):
        return None
    return tuple(resolved)


def _build_segment_g2pw_partial_result(seg_cut, seg: str):
    partial_result = [None] * len(seg)
    cursor = 0
    has_override = False
    for word, pos in seg_cut:
        word_len = len(word)
        if pos != "eng":
            phrase_override = _get_phrase_level_pinyin_override(word)
            if phrase_override is not None:
                partial_result[cursor : cursor + word_len] = phrase_override
                has_override = True
        cursor += word_len

    if cursor != len(seg) or not has_override:
        return None
    return partial_result


must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
not_erhua = {
    "虐儿",
    "为儿",
    "护儿",
    "瞒儿",
    "救儿",
    "替儿",
    "有儿",
    "一儿",
    "我儿",
    "俺儿",
    "妻儿",
    "拐儿",
    "聋儿",
    "乞儿",
    "患儿",
    "幼儿",
    "孤儿",
    "婴儿",
    "婴幼儿",
    "连体儿",
    "脑瘫儿",
    "流浪儿",
    "体弱儿",
    "混血儿",
    "蜜雪儿",
    "舫儿",
    "祖儿",
    "美儿",
    "应采儿",
    "可儿",
    "侄儿",
    "孙儿",
    "侄孙儿",
    "女儿",
    "男儿",
    "红孩儿",
    "花儿",
    "虫儿",
    "马儿",
    "鸟儿",
    "猪儿",
    "猫儿",
    "狗儿",
    "少儿",
}


def _merge_erhua(initials: list[str], finals: list[str], word: str, pos: str) -> list[list[str]]:
    """
    Do erhub.
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"

    # 发音
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals

    # "……" 等情况直接返回
    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    # 与前一个字发同音
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if (
            i == len(finals) - 1
            and word[i] == "儿"
            and phn in {"er2", "er5"}
            and word[-2:] not in not_erhua
            and new_finals
        ):
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


def _map_initial_final_to_phones(c, v, seg):
    raw_pinyin = c + v
    if c == v:
        assert c in punctuation
        return [c]

    v_without_tone = v[:-1]
    tone = v[-1]

    pinyin = c + v_without_tone
    assert tone in "12345"

    if c:
        v_rep_map = {
            "uei": "ui",
            "iou": "iu",
            "uen": "un",
        }
        if v_without_tone in v_rep_map.keys():
            pinyin = c + v_rep_map[v_without_tone]
    else:
        pinyin_rep_map = {
            "ing": "ying",
            "i": "yi",
            "in": "yin",
            "u": "wu",
        }
        if pinyin in pinyin_rep_map.keys():
            pinyin = pinyin_rep_map[pinyin]
        else:
            single_rep_map = {
                "v": "yu",
                "e": "e",
                "i": "y",
                "u": "w",
            }
            if pinyin[0] in single_rep_map.keys():
                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

    assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
    new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
    new_v = new_v + tone
    return [new_c, new_v]


def _g2p(segments, return_phone_units: bool = False):
    phones_list = []
    word2ph = []
    phone_units = []
    g2pw_batch_results = []
    g2pw_batch_cursor = 0
    char_cursor = 0
    processed_segments = [re.sub("[a-zA-Z]+", "", seg) for seg in segments]
    seg_cuts = []
    g2pw_partial_results = []
    for seg in processed_segments:
        seg_cut = psg.lcut(seg)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        seg_cuts.append(seg_cut)
        if is_g2pw and seg:
            g2pw_partial_results.append(_build_segment_g2pw_partial_result(seg_cut, seg))
        else:
            g2pw_partial_results.append(None)
    if is_g2pw:
        batch_inputs = [seg for seg in processed_segments if seg]
        batch_partial_results = [result for seg, result in zip(processed_segments, g2pw_partial_results) if seg]
        g2pw_batch_results = g2pw(batch_inputs, partial_results=batch_partial_results) if batch_inputs else []

    for seg, seg_cut in zip(processed_segments, seg_cuts):
        pinyins = []
        if seg:
            if is_g2pw:
                pinyins = g2pw_batch_results[g2pw_batch_cursor]
                g2pw_batch_cursor += 1
            else:
                pinyins = None

        pre_word_length = 0
        for word, pos in seg_cut:
            if pos == "eng":
                pre_word_length += len(word)
                continue

            if is_g2pw:
                sub_initials = []
                sub_finals = []
                now_word_length = pre_word_length + len(word)
                word_pinyins = pinyins[pre_word_length:now_word_length]
                word_pinyins = correct_pronunciation(word, word_pinyins)
                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)
                pre_word_length = now_word_length
            else:
                sub_initials, sub_finals = _get_initials_finals(word)

            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)

            unit_phones = []
            unit_type = "punct" if all(char in punctuation for char in word) else "word"
            for c, v in zip(sub_initials, sub_finals):
                phone = _map_initial_final_to_phones(c, v, seg)
                unit_phones.extend(phone)
                phones_list.extend(phone)
                word2ph.append(len(phone))

            if return_phone_units and word:
                phone_units.append(
                    {
                        "unit_type": unit_type,
                        "text": word,
                        "norm_text": word,
                        "pos": pos,
                        "phones": unit_phones,
                        "char_start": int(char_cursor),
                        "char_end": int(char_cursor + len(word)),
                    }
                )
            char_cursor += len(word)

    if return_phone_units:
        return phones_list, word2ph, finalize_phone_units(phone_units)
    return phones_list, word2ph


def replace_punctuation_with_en(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5A-Za-z" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def text_normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    sentences = text_normalizer.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


if __name__ == "__main__":
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = "呣呣呣～就是…大人的鼹鼠党吧？"
    text = "你好"
    text = text_normalize(text)
    print(g2p(text))


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试..."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
