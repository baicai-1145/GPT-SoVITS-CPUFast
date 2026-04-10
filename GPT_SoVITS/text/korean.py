# reference: https://github.com/ORI-Muchim/MB-iSTFT-VITS-Korean/blob/main/text/korean.py

import re
import sys
import threading
import types
from jamo import h2j, j2hcj
import ko_pron

import importlib
import importlib.util
import os
from text.phone_units import finalize_phone_units, flatten_phone_units


class _LazyCmuDict:
    def __init__(self):
        self._cmu = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._cmu is not None:
            return self._cmu
        with self._lock:
            if self._cmu is not None:
                return self._cmu
            old_nltk = sys.modules.pop("nltk", None)
            old_corpus = sys.modules.pop("nltk.corpus", None)
            old_cmudict = sys.modules.pop("nltk.corpus.cmudict", None)
            try:
                nltk = importlib.import_module("nltk")
                from nltk.corpus import cmudict

                try:
                    nltk.data.find("corpora/cmudict.zip")
                except LookupError:
                    nltk.download("cmudict")
                self._cmu = cmudict.dict()
            finally:
                if old_nltk is not None:
                    sys.modules["nltk"] = old_nltk
                if old_corpus is not None:
                    sys.modules["nltk.corpus"] = old_corpus
                if old_cmudict is not None:
                    sys.modules["nltk.corpus.cmudict"] = old_cmudict
            return self._cmu

    def __contains__(self, key):
        return key in self._ensure()

    def __getitem__(self, key):
        return self._ensure()[key]

    def get(self, key, default=None):
        return self._ensure().get(key, default)


def _install_lazy_nltk_stub():
    if "nltk" in sys.modules:
        return None
    nltk_stub = types.ModuleType("nltk")

    class _Data:
        @staticmethod
        def find(path):
            return path

    nltk_stub.data = _Data()
    nltk_stub.download = lambda name: True

    corpus_stub = types.ModuleType("nltk.corpus")
    cmudict_stub = types.ModuleType("nltk.corpus.cmudict")
    cmudict_stub.dict = lambda: _LazyCmuDict()
    corpus_stub.cmudict = cmudict_stub
    nltk_stub.corpus = corpus_stub

    sys.modules["nltk"] = nltk_stub
    sys.modules["nltk.corpus"] = corpus_stub
    sys.modules["nltk.corpus.cmudict"] = cmudict_stub
    return ("nltk", "nltk.corpus", "nltk.corpus.cmudict")


def _cleanup_lazy_nltk_stub(installed_names):
    if not installed_names:
        return
    for name in installed_names:
        sys.modules.pop(name, None)


_stubbed_nltk_modules = _install_lazy_nltk_stub()
try:
    from g2pk2 import G2p
finally:
    _cleanup_lazy_nltk_stub(_stubbed_nltk_modules)

# 防止win下无法读取模型
if os.name == "nt":

    class win_G2p(G2p):
        def check_mecab(self):
            super().check_mecab()
            spam_spec = importlib.util.find_spec("eunjeon")
            non_found = spam_spec is None
            if non_found:
                print("you have to install eunjeon. install it...")
            else:
                installpath = spam_spec.submodule_search_locations[0]
                if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                    import sys
                    from eunjeon import Mecab as _Mecab

                    class Mecab(_Mecab):
                        def get_dicpath(installpath):
                            if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                                import shutil

                                python_dir = os.getcwd()
                                if installpath[: len(python_dir)].upper() == python_dir.upper():
                                    dicpath = os.path.join(os.path.relpath(installpath, python_dir), "data", "mecabrc")
                                else:
                                    if not os.path.exists("TEMP"):
                                        os.mkdir("TEMP")
                                    if not os.path.exists(os.path.join("TEMP", "ko")):
                                        os.mkdir(os.path.join("TEMP", "ko"))
                                    if os.path.exists(os.path.join("TEMP", "ko", "ko_dict")):
                                        shutil.rmtree(os.path.join("TEMP", "ko", "ko_dict"))

                                    shutil.copytree(
                                        os.path.join(installpath, "data"), os.path.join("TEMP", "ko", "ko_dict")
                                    )
                                    dicpath = os.path.join("TEMP", "ko", "ko_dict", "mecabrc")
                            else:
                                dicpath = os.path.abspath(os.path.join(installpath, "data/mecabrc"))
                            return dicpath

                        def __init__(self, dicpath=get_dicpath(installpath)):
                            super().__init__(dicpath=dicpath)

                    sys.modules["eunjeon"].Mecab = Mecab

    G2p = win_G2p


from text.symbols2 import symbols

# This is a list of Korean classifiers preceded by pure Korean numerals.
_korean_classifiers = (
    "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통"
)

# List of (hangul, hangul divided) pairs:
_hangul_divided = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        # ('ㄳ', 'ㄱㅅ'),   # g2pk2, A Syllable-ending Rule
        # ('ㄵ', 'ㄴㅈ'),
        # ('ㄶ', 'ㄴㅎ'),
        # ('ㄺ', 'ㄹㄱ'),
        # ('ㄻ', 'ㄹㅁ'),
        # ('ㄼ', 'ㄹㅂ'),
        # ('ㄽ', 'ㄹㅅ'),
        # ('ㄾ', 'ㄹㅌ'),
        # ('ㄿ', 'ㄹㅍ'),
        # ('ㅀ', 'ㄹㅎ'),
        # ('ㅄ', 'ㅂㅅ'),
        ("ㅘ", "ㅗㅏ"),
        ("ㅙ", "ㅗㅐ"),
        ("ㅚ", "ㅗㅣ"),
        ("ㅝ", "ㅜㅓ"),
        ("ㅞ", "ㅜㅔ"),
        ("ㅟ", "ㅜㅣ"),
        ("ㅢ", "ㅡㅣ"),
        ("ㅑ", "ㅣㅏ"),
        ("ㅒ", "ㅣㅐ"),
        ("ㅕ", "ㅣㅓ"),
        ("ㅖ", "ㅣㅔ"),
        ("ㅛ", "ㅣㅗ"),
        ("ㅠ", "ㅣㅜ"),
    ]
]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "에이"),
        ("b", "비"),
        ("c", "시"),
        ("d", "디"),
        ("e", "이"),
        ("f", "에프"),
        ("g", "지"),
        ("h", "에이치"),
        ("i", "아이"),
        ("j", "제이"),
        ("k", "케이"),
        ("l", "엘"),
        ("m", "엠"),
        ("n", "엔"),
        ("o", "오"),
        ("p", "피"),
        ("q", "큐"),
        ("r", "아르"),
        ("s", "에스"),
        ("t", "티"),
        ("u", "유"),
        ("v", "브이"),
        ("w", "더블유"),
        ("x", "엑스"),
        ("y", "와이"),
        ("z", "제트"),
    ]
]

# List of (ipa, lazy ipa) pairs:
_ipa_to_lazy_ipa = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("t͡ɕ", "ʧ"),
        ("d͡ʑ", "ʥ"),
        ("ɲ", "n^"),
        ("ɕ", "ʃ"),
        ("ʷ", "w"),
        ("ɭ", "l`"),
        ("ʎ", "ɾ"),
        ("ɣ", "ŋ"),
        ("ɰ", "ɯ"),
        ("ʝ", "j"),
        ("ʌ", "ə"),
        ("ɡ", "g"),
        ("\u031a", "#"),
        ("\u0348", "="),
        ("\u031e", ""),
        ("\u0320", ""),
        ("\u0339", ""),
    ]
]


def fix_g2pk2_error(text):
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i : i + 3] == "ㅇㅡㄹ" or text[i : i + 3] == "ㄹㅡㄹ") and text[i + 3] == " " and text[i + 4] == "ㄹ":
            new_text += text[i : i + 3] + " " + "ㄴ"
            i += 5
        else:
            new_text += text[i]
            i += 1

    new_text += text[i:]
    return new_text


def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text


def divide_hangul(text):
    text = j2hcj(h2j(text))
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text


def hangul_number(num, sino=True):
    """Reference https://github.com/Kyubyong/g2pK"""
    num = re.sub(",", "", num)

    if num == "0":
        return "영"
    if not sino and num == "20":
        return "스무"

    digits = "123456789"
    names = "일이삼사오육칠팔구"
    digit2name = {d: n for d, n in zip(digits, names)}

    modifiers = "한 두 세 네 다섯 여섯 일곱 여덟 아홉"
    decimals = "열 스물 서른 마흔 쉰 예순 일흔 여든 아흔"
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, "")
            elif i == 1:
                name = digit2name.get(digit, "") + "십"
                name = name.replace("일십", "십")
        else:
            if i == 0:
                name = digit2mod.get(digit, "")
            elif i == 1:
                name = digit2dec.get(digit, "")
        if digit == "0":
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)) :]
                if "".join(last_three) == "":
                    spelledout.append("")
                    continue
            else:
                spelledout.append("")
                continue
        if i == 2:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 3:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 4:
            name = digit2name.get(digit, "") + "만"
            name = name.replace("일만", "만")
        elif i == 5:
            name = digit2name.get(digit, "") + "십"
            name = name.replace("일십", "십")
        elif i == 6:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 7:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 8:
            name = digit2name.get(digit, "") + "억"
        elif i == 9:
            name = digit2name.get(digit, "") + "십"
        elif i == 10:
            name = digit2name.get(digit, "") + "백"
        elif i == 11:
            name = digit2name.get(digit, "") + "천"
        elif i == 12:
            name = digit2name.get(digit, "") + "조"
        elif i == 13:
            name = digit2name.get(digit, "") + "십"
        elif i == 14:
            name = digit2name.get(digit, "") + "백"
        elif i == 15:
            name = digit2name.get(digit, "") + "천"
        spelledout.append(name)
    return "".join(elem for elem in spelledout)


def number_to_hangul(text):
    """Reference https://github.com/Kyubyong/g2pK"""
    tokens = set(re.findall(r"(\d[\d,]*)([\uac00-\ud71f]+)", text))
    for token in tokens:
        num, classifier = token
        if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
            spelledout = hangul_number(num, sino=False)
        else:
            spelledout = hangul_number(num, sino=True)
        text = text.replace(f"{num}{classifier}", f"{spelledout}{classifier}")
    # digit by digit for remaining digits
    digits = "0123456789"
    names = "영일이삼사오육칠팔구"
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text


def korean_to_lazy_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = re.sub("[\uac00-\ud7af]+", lambda x: ko_pron.romanise(x.group(0), "ipa").split("] ~ [")[0], text)
    for regex, replacement in _ipa_to_lazy_ipa:
        text = re.sub(regex, replacement, text)
    return text


_g2p = G2p()


def korean_to_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = _g2p(text)
    text = fix_g2pk2_error(text)
    text = korean_to_lazy_ipa(text)
    return text.replace("ʧ", "tʃ").replace("ʥ", "dʑ")


def post_replace_ph(ph):
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
        " ": "空",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "停"
    return ph


def _is_korean_separator(char):
    if char.isspace():
        return "space"
    if char in {"：", "；", "，", "。", "！", "？", "\n", "·", "、", ".", ",", "!", "?", ";", ":"}:
        return "punct"
    return "word"


def _split_korean_units(text):
    if not text:
        return []
    units = []
    cursor = 0
    while cursor < len(text):
        kind = _is_korean_separator(text[cursor])
        end = cursor + 1
        if kind != "word":
            while end < len(text) and _is_korean_separator(text[end]) == kind:
                end += 1
        else:
            while end < len(text) and _is_korean_separator(text[end]) == "word":
                end += 1
        units.append((kind, text[cursor:end]))
        cursor = end
    return units


def _transform_g2p_text(text):
    text = latin_to_hangul(text)
    text = _g2p(text)
    text = divide_hangul(text)
    text = fix_g2pk2_error(text)
    text = re.sub(r"([\u3131-\u3163])$", r"\1.", text)
    return text


def g2p(text):
    return flatten_phone_units(g2p_with_phone_units(text)[1])


def g2p_with_phone_units(text):
    transformed = _transform_g2p_text(text)
    source_units = _split_korean_units(text)
    transformed_units = _split_korean_units(transformed)

    if len(source_units) + 1 == len(transformed_units):
        extra_kind, extra_text = transformed_units[-1]
        if extra_kind == "punct" and (not source_units or source_units[-1] != (extra_kind, extra_text)):
            source_units = source_units + [(extra_kind, extra_text)]

    if len(source_units) != len(transformed_units):
        raise RuntimeError(
            f"Korean unit count mismatch: source={len(source_units)} transformed={len(transformed_units)}"
        )

    units = []
    for (source_kind, source_text), (transformed_kind, transformed_text) in zip(source_units, transformed_units):
        if source_kind != transformed_kind:
            raise RuntimeError(
                f"Korean unit type mismatch: source={source_kind}:{source_text!r} "
                f"transformed={transformed_kind}:{transformed_text!r}"
            )
        units.append(
            {
                "unit_type": source_kind,
                "text": source_text,
                "norm_text": transformed_text,
                "phones": [post_replace_ph(ch) for ch in transformed_text],
            }
        )

    units = finalize_phone_units(units)
    return flatten_phone_units(units), units


if __name__ == "__main__":
    text = "안녕하세요"
    print(g2p(text))
