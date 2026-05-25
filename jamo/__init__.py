"""Minimal local replacement for the python-jamo APIs used by Korean G2P."""

from itertools import chain


_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_LEAD_BASE = 0x1100
_VOWEL_BASE = 0x1161
_TAIL_BASE = 0x11A7

_LEAD_TO_HCJ = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWEL_TO_HCJ = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_TAIL_TO_HCJ = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

_JAMO_TO_HCJ = {}
_JAMO_TO_HCJ.update({chr(_LEAD_BASE + i): ch for i, ch in enumerate(_LEAD_TO_HCJ)})
_JAMO_TO_HCJ.update({chr(_VOWEL_BASE + i): ch for i, ch in enumerate(_VOWEL_TO_HCJ)})
_JAMO_TO_HCJ.update({chr(_TAIL_BASE + i + 1): ch for i, ch in enumerate(_TAIL_TO_HCJ)})

_HCJ_TO_LEAD = {ch: chr(_LEAD_BASE + i) for i, ch in enumerate(_LEAD_TO_HCJ)}
_HCJ_TO_VOWEL = {ch: chr(_VOWEL_BASE + i) for i, ch in enumerate(_VOWEL_TO_HCJ)}
_HCJ_TO_TAIL = {ch: chr(_TAIL_BASE + i + 1) for i, ch in enumerate(_TAIL_TO_HCJ)}


def _hangul_char_to_jamo(char):
    code = ord(char)
    if not (_HANGUL_BASE <= code <= _HANGUL_END):
        return char

    syllable_index = code - _HANGUL_BASE
    lead_index = syllable_index // 588
    vowel_index = (syllable_index % 588) // 28
    tail_index = syllable_index % 28

    decomposed = [chr(_LEAD_BASE + lead_index), chr(_VOWEL_BASE + vowel_index)]
    if tail_index:
        decomposed.append(chr(_TAIL_BASE + tail_index))
    return decomposed


def hangul_to_jamo(text):
    return chain.from_iterable(_hangul_char_to_jamo(char) for char in text)


def h2j(text):
    return "".join(hangul_to_jamo(text))


def jamo_to_hcj(text):
    return (_JAMO_TO_HCJ.get(char, char) for char in text)


def j2hcj(text):
    return "".join(jamo_to_hcj(text))


def _as_char(value):
    if isinstance(value, int):
        return chr(value)
    return value


def _lead_index(char):
    char = _as_char(char)
    if char in _HCJ_TO_LEAD:
        char = _HCJ_TO_LEAD[char]
    code = ord(char)
    if not (_LEAD_BASE <= code <= _LEAD_BASE + 18):
        raise ValueError(f"Invalid Hangul lead jamo: {char!r}")
    return code - _LEAD_BASE


def _vowel_index(char):
    char = _as_char(char)
    if char in _HCJ_TO_VOWEL:
        char = _HCJ_TO_VOWEL[char]
    code = ord(char)
    if not (_VOWEL_BASE <= code <= _VOWEL_BASE + 20):
        raise ValueError(f"Invalid Hangul vowel jamo: {char!r}")
    return code - _VOWEL_BASE


def _tail_index(char):
    if not char or char == 0:
        return 0
    char = _as_char(char)
    if char in _HCJ_TO_TAIL:
        char = _HCJ_TO_TAIL[char]
    code = ord(char)
    if not (_TAIL_BASE + 1 <= code <= _TAIL_BASE + 27):
        raise ValueError(f"Invalid Hangul tail jamo: {char!r}")
    return code - _TAIL_BASE


def j2h(lead, vowel, tail=0):
    code = _HANGUL_BASE + (_lead_index(lead) * 21 + _vowel_index(vowel)) * 28 + _tail_index(tail)
    return chr(code)


def jamo_to_hangul(lead, vowel, tail=0):
    return j2h(lead, vowel, tail)
