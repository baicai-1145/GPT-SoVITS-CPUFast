from __future__ import annotations

from typing import Iterable


def finalize_phone_units(units: Iterable[dict]) -> list[dict]:
    finalized = []
    cursor = 0
    for raw_unit in units:
        phones = list(raw_unit.get("phones", []))
        unit = dict(raw_unit)
        unit["phones"] = phones
        unit["phone_count"] = int(len(phones))
        unit["phone_start"] = int(cursor)
        cursor += len(phones)
        unit["phone_end"] = int(cursor)
        finalized.append(unit)
    return finalized


def flatten_phone_units(units: Iterable[dict]) -> list:
    phones = []
    for unit in units:
        phones.extend(unit.get("phones", []))
    return phones


def build_char_phone_units(norm_text: str, word2ph: list[int], phones: list | None = None) -> list[dict]:
    if len(norm_text) != len(word2ph):
        raise ValueError(f"char/word2ph length mismatch: text={len(norm_text)} word2ph={len(word2ph)}")

    units = []
    phone_cursor = 0
    for char_index, (char, phone_count) in enumerate(zip(norm_text, word2ph)):
        phone_count = int(phone_count)
        if phone_count < 0:
            raise ValueError(f"Negative phone_count at char_index={char_index}: {phone_count}")
        unit_phones = []
        if phones is not None:
            unit_phones = list(phones[phone_cursor : phone_cursor + phone_count])
        units.append(
            {
                "unit_type": "char",
                "text": char,
                "norm_text": char,
                "phones": unit_phones,
                "char_start": int(char_index),
                "char_end": int(char_index + 1),
                "phone_start": int(phone_cursor),
                "phone_end": int(phone_cursor + phone_count),
                "phone_count": int(phone_count),
            }
        )
        phone_cursor += phone_count
    return units
