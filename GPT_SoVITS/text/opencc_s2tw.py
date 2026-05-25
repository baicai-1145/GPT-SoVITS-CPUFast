import os
from functools import lru_cache
from typing import Dict, Tuple


_ASSET_DIR = os.path.join(os.path.dirname(__file__), "opencc_s2tw_assets")


def _load_dict(path: str, first_only: bool) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            key, values = line.split("\t", 1)
            value = values.split(" ")[0] if first_only else values
            data[key] = value
    return data


def _build_prefixes(mapping: Dict[str, str]) -> Tuple[set[str], int]:
    prefixes = set()
    max_len = 1
    for key in mapping:
        max_len = max(max_len, len(key))
        for i in range(1, len(key)):
            prefixes.add(key[:i])
    return prefixes, max_len


@lru_cache(maxsize=1)
def _load_assets():
    phrases = _load_dict(os.path.join(_ASSET_DIR, "STPhrases.txt"), first_only=True)
    chars = _load_dict(os.path.join(_ASSET_DIR, "STCharacters.txt"), first_only=True)
    variants = _load_dict(os.path.join(_ASSET_DIR, "TWVariants.txt"), first_only=True)
    phrase_prefixes, max_phrase_len = _build_prefixes(phrases)
    return phrases, chars, variants, phrase_prefixes, max_phrase_len


def _convert_phrases_and_chars(text: str) -> str:
    phrases, chars, _variants, phrase_prefixes, max_phrase_len = _load_assets()
    output = []
    i = 0
    while i < len(text):
        matched = None
        matched_len = 0
        limit = min(len(text), i + max_phrase_len)
        probe = i + 1
        while probe <= limit:
            chunk = text[i:probe]
            if chunk in phrases:
                matched = phrases[chunk]
                matched_len = len(chunk)
            if chunk not in phrase_prefixes:
                break
            probe += 1

        if matched is not None:
            output.append(matched)
            i += matched_len
            continue

        char = text[i]
        output.append(chars.get(char, char))
        i += 1
    return "".join(output)


def simplified_to_traditional_tw(text: str) -> str:
    """Local replacement for OpenCC s2tw used by the G2PW frontend."""
    _phrases, _chars, variants, _phrase_prefixes, _max_phrase_len = _load_assets()
    converted = _convert_phrases_and_chars(text)
    return "".join(variants.get(char, char) for char in converted)
