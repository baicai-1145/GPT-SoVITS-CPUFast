# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
import os
import hashlib

try:
    import pyopenjtalk

    current_file_path = os.path.dirname(__file__)

    # 防止win下无法读取模型
    if os.name == "nt":
        python_dir = os.getcwd()
        OPEN_JTALK_DICT_DIR = pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8")
        if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", OPEN_JTALK_DICT_DIR)):
            if OPEN_JTALK_DICT_DIR[: len(python_dir)].upper() == python_dir.upper():
                OPEN_JTALK_DICT_DIR = os.path.join(os.path.relpath(OPEN_JTALK_DICT_DIR, python_dir))
            else:
                import shutil

                if not os.path.exists("TEMP"):
                    os.mkdir("TEMP")
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if os.path.exists(os.path.join("TEMP", "ja", "open_jtalk_dic")):
                    shutil.rmtree(os.path.join("TEMP", "ja", "open_jtalk_dic"))
                shutil.copytree(
                    pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"),
                    os.path.join("TEMP", "ja", "open_jtalk_dic"),
                )
                OPEN_JTALK_DICT_DIR = os.path.join("TEMP", "ja", "open_jtalk_dic")
            pyopenjtalk.OPEN_JTALK_DICT_DIR = OPEN_JTALK_DICT_DIR.encode("utf-8")

        if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", current_file_path)):
            if current_file_path[: len(python_dir)].upper() == python_dir.upper():
                current_file_path = os.path.join(os.path.relpath(current_file_path, python_dir))
            else:
                if not os.path.exists("TEMP"):
                    os.mkdir("TEMP")
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if not os.path.exists(os.path.join("TEMP", "ja", "ja_userdic")):
                    os.mkdir(os.path.join("TEMP", "ja", "ja_userdic"))
                    shutil.copyfile(
                        os.path.join(current_file_path, "ja_userdic", "userdict.csv"),
                        os.path.join("TEMP", "ja", "ja_userdic", "userdict.csv"),
                    )
                current_file_path = os.path.join("TEMP", "ja")

    def get_hash(fp: str) -> str:
        hash_md5 = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    USERDIC_CSV_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.csv")
    USERDIC_BIN_PATH = os.path.join(current_file_path, "ja_userdic", "user.dict")
    USERDIC_HASH_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.md5")
    # 如果没有用户词典，就生成一个；如果有，就检查md5，如果不一样，就重新生成
    if os.path.exists(USERDIC_CSV_PATH):
        if (
            not os.path.exists(USERDIC_BIN_PATH)
            or get_hash(USERDIC_CSV_PATH) != open(USERDIC_HASH_PATH, "r", encoding="utf-8").read()
        ):
            pyopenjtalk.mecab_dict_index(USERDIC_CSV_PATH, USERDIC_BIN_PATH)
            with open(USERDIC_HASH_PATH, "w", encoding="utf-8") as f:
                f.write(get_hash(USERDIC_CSV_PATH))

    if os.path.exists(USERDIC_BIN_PATH):
        pyopenjtalk.update_global_jtalk_with_user_dict(USERDIC_BIN_PATH)
except Exception:
    # print(e)
    import pyopenjtalk

    # failed to load user dictionary, ignore.
    pass


from text.symbols import punctuation
from text.phone_units import finalize_phone_units, flatten_phone_units

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]


# List of (consonant, sokuon) pairs:
_real_sokuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"Q([↑↓]*[kg])", r"k#\1"),
        (r"Q([↑↓]*[tdjʧ])", r"t#\1"),
        (r"Q([↑↓]*[sʃ])", r"s\1"),
        (r"Q([↑↓]*[pb])", r"p#\1"),
    ]
]

# List of (consonant, hatsuon) pairs:
_real_hatsuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"N([↑↓]*[pbm])", r"m\1"),
        (r"N([↑↓]*[ʧʥj])", r"n^\1"),
        (r"N([↑↓]*[tdn])", r"n\1"),
        (r"N([↑↓]*[kg])", r"ŋ\1"),
    ]
]

_prosody_marks = {"^", "$", "?", "_", "#", "[", "]"}


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
    }

    if ph in rep_map.keys():
        ph = rep_map[ph]
    return ph


def _normalize_alignment_phone(ph):
    if ph in {"A", "E", "I", "O", "U"}:
        return ph.lower()
    return ph


def _phones_equivalent(lhs, rhs):
    return _normalize_alignment_phone(lhs) == _normalize_alignment_phone(rhs)


def _is_prosody_mark(ph):
    return ph in _prosody_marks


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text, with_prosody=False):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    text = symbols_to_japanese(text)
    # English words to lower case, should have no influence on japanese words.
    text = text.lower()
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = []
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if with_prosody:
                text += pyopenjtalk_g2p_prosody(sentence)[1:-1]
            else:
                p = pyopenjtalk.g2p(sentence)
                text += p.split(" ")

        if i < len(marks):
            if marks[i] == " ":  # 防止意外的UNK
                continue
            text += [marks[i].replace(" ", "")]
    return text


def _frontend_word_phone_candidates(word_info):
    raw_candidates = []
    for key in ("string", "pron", "read"):
        value = (word_info.get(key) or "").replace("’", "").replace("'", "")
        if not value or value == "*" or value in raw_candidates:
            continue
        raw_candidates.append(value)

    candidates = []
    for value in raw_candidates:
        raw = pyopenjtalk.g2p(value)
        phones = [post_replace_ph(item) for item in raw.split(" ") if item]
        if phones and phones not in candidates:
            candidates.append(phones)
    return candidates


def _align_word_candidate(full_tokens, cursor, candidate_phones):
    probe = cursor
    matched = 0
    unit_tokens = []
    while probe < len(full_tokens) and matched < len(candidate_phones):
        current = full_tokens[probe]
        if _phones_equivalent(current, candidate_phones[matched]):
            unit_tokens.append(current)
            matched += 1
            probe += 1
            continue
        if _is_prosody_mark(current):
            unit_tokens.append(current)
            probe += 1
            continue
        return None
    if matched != len(candidate_phones):
        return None
    return unit_tokens, probe


def _build_word_unit(word_info, unit_tokens):
    return {
        "unit_type": "word",
        "text": word_info.get("string", ""),
        "norm_text": (word_info.get("pron") or word_info.get("read") or word_info.get("string") or "").replace("’", ""),
        "phones": unit_tokens,
    }


def _build_prosody_units(full_tokens, start, end):
    return [
        {
            "unit_type": "prosody",
            "text": full_tokens[idx],
            "norm_text": full_tokens[idx],
            "phones": [full_tokens[idx]],
        }
        for idx in range(start, end)
    ]


def _align_frontend_words(frontend, full_tokens, word_index, cursor):
    if word_index >= len(frontend):
        trailing_end = cursor
        while trailing_end < len(full_tokens) and _is_prosody_mark(full_tokens[trailing_end]):
            trailing_end += 1
        if trailing_end != len(full_tokens):
            return None
        return _build_prosody_units(full_tokens, cursor, trailing_end)

    word_info = frontend[word_index]
    candidates = _frontend_word_phone_candidates(word_info)
    for candidate_phones in candidates:
        aligned = _align_word_candidate(full_tokens, cursor, candidate_phones)
        if aligned is None:
            continue
        unit_tokens, next_cursor = aligned
        boundary_end = next_cursor
        while boundary_end < len(full_tokens) and _is_prosody_mark(full_tokens[boundary_end]):
            boundary_end += 1
        rest = _align_frontend_words(frontend, full_tokens, word_index + 1, boundary_end)
        if rest is None:
            continue
        return [_build_word_unit(word_info, unit_tokens)] + _build_prosody_units(full_tokens, next_cursor, boundary_end) + rest

    if word_index == len(frontend) - 1 and cursor < len(full_tokens):
        return [_build_word_unit(word_info, full_tokens[cursor:])]
    return None


def _sentence_phone_units(sentence, with_prosody=True):
    frontend = pyopenjtalk.run_frontend(sentence)
    full_tokens = [post_replace_ph(item) for item in preprocess_jap(sentence, with_prosody=with_prosody)]
    units = _align_frontend_words(frontend, full_tokens, 0, 0)
    if units is None:
        return [
            {
                "unit_type": "word_group",
                "text": sentence,
                "norm_text": sentence,
                "phones": full_tokens,
            }
        ]
    return units


def text_normalize(text):
    # todo: jap text normalize

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)
    return text


# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def g2p(norm_text, with_prosody=True):
    return flatten_phone_units(g2p_with_phone_units(norm_text, with_prosody)[1])


def g2p_with_phone_units(norm_text, with_prosody=True):
    text = symbols_to_japanese(norm_text)
    text = text.lower()
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)

    units = []
    for idx, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            units.extend(_sentence_phone_units(sentence, with_prosody=with_prosody))
        if idx < len(marks):
            mark = marks[idx]
            if mark == " ":
                continue
            cleaned_mark = mark.replace(" ", "")
            units.append(
                {
                    "unit_type": "punct",
                    "text": mark,
                    "norm_text": cleaned_mark,
                    "phones": [post_replace_ph(cleaned_mark)],
                }
            )

    units = finalize_phone_units(units)
    return flatten_phone_units(units), units


if __name__ == "__main__":
    phones = g2p("Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
    print(phones)
