"""Cached pypinyin hot-path wrappers.

tone_sandhi's merge passes call lazy_pinyin(FINALS_TONE3) 2+ times per word,
and the phrase-override path calls the heavier pinyin(). Words repeat heavily
across a sentence/corpus, so memoize the (word, style) -> result mapping.

Returns are fresh lists (copies of cached tuples): callers may mutate freely.
Semantics identical to the uncached calls.
"""
from functools import lru_cache

from pypinyin import Style
from pypinyin import lazy_pinyin as _lazy_pinyin
from pypinyin import pinyin as _pinyin


@lru_cache(maxsize=65536)
def _finals_tone3(word: str):
    return tuple(_lazy_pinyin(word, neutral_tone_with_five=True,
                              style=Style.FINALS_TONE3))


@lru_cache(maxsize=65536)
def _initials(word: str):
    return tuple(_lazy_pinyin(word, neutral_tone_with_five=True,
                              style=Style.INITIALS))


@lru_cache(maxsize=65536)
def _tone3(word: str):
    return tuple(tuple(x) for x in _pinyin(word, neutral_tone_with_five=True,
                                           style=Style.TONE3))


def cached_lazy_pinyin(*args, **kwargs):
    style = kwargs.get("style")
    if kwargs.get("neutral_tone_with_five") and len(args) == 1:
        if style == Style.FINALS_TONE3:
            return list(_finals_tone3(args[0]))
        if style == Style.INITIALS:
            return list(_initials(args[0]))
    return _lazy_pinyin(*args, **kwargs)


def cached_pinyin(*args, **kwargs):
    if (kwargs.get("neutral_tone_with_five") and len(args) == 1
            and kwargs.get("style") == Style.TONE3):
        return [list(x) for x in _tone3(args[0])]
    return _pinyin(*args, **kwargs)
