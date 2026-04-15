import ast
import os
import pickle

from text.zh_normalization.char_convert import tranditional_to_simplified

current_file_path = os.path.dirname(__file__)
CACHE_PATH = os.path.join(current_file_path, "polyphonic.pickle")
PP_DICT_PATH = os.path.join(current_file_path, "polyphonic.rep")
PP_FIX_DICT_PATH = os.path.join(current_file_path, "polyphonic-fix.rep")
PHRASE_OVERRIDE_PATH = os.path.join(current_file_path, "phrase_overrides.pkl")
EXTRA_PHRASE_OVERRIDE_PATHS = [
    os.path.join(current_file_path, "phrase_overrides_zdic_strict.pkl"),
    os.path.join(current_file_path, "phrase_overrides_hwxnet_idiom.pkl"),
]
PHRASE_OVERRIDE_CACHE_PATH = os.path.join(current_file_path, "phrase_overrides_bundle.pickle")
PHRASE_OVERRIDE_CACHE_VERSION = 2


def cache_dict(polyphonic_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(polyphonic_dict, pickle_file)


def _read_rep_file(file_path):
    data = {}
    with open(file_path, encoding="utf-8") as f:
        line = f.readline()
        while line:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                key, value_str = line.split(":", 1)
                data[key.strip()] = ast.literal_eval(value_str.strip())
            line = f.readline()
    return data


def _read_pickle_file(file_path):
    with open(file_path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    if not isinstance(data, dict):
        raise TypeError(f"{file_path} does not contain a dict")
    return data


def _read_phrase_override_file(file_path):
    _, ext = os.path.splitext(file_path)
    if ext == ".pkl":
        return _read_pickle_file(file_path)
    if ext == ".rep":
        return _read_rep_file(file_path)
    raise ValueError(f"Unsupported phrase override file format: {file_path}")


def _rep_file_signature(file_path):
    if not os.path.exists(file_path):
        return (file_path, None, None)
    stat = os.stat(file_path)
    return (file_path, stat.st_size, stat.st_mtime_ns)


def _build_phrase_override_signature():
    return (
        PHRASE_OVERRIDE_CACHE_VERSION,
        tuple(_rep_file_signature(path) for path in [*EXTRA_PHRASE_OVERRIDE_PATHS, PHRASE_OVERRIDE_PATH]),
    )


def _is_valid_phrase_override(word, pronunciations):
    return isinstance(pronunciations, list) and len(word) > 1 and len(pronunciations) == len(word)


def _merge_phrase_override_entries(target, metadata, entries, priority):
    for word, pronunciations in entries.items():
        if not _is_valid_phrase_override(word, pronunciations):
            continue
        target[word] = pronunciations
        metadata[word] = (priority, True)

        simplified_word = tranditional_to_simplified(word)
        if simplified_word == word or len(simplified_word) != len(word):
            continue
        prev_meta = metadata.get(simplified_word)
        if prev_meta is not None and prev_meta[1]:
            continue
        if prev_meta is not None and prev_meta[0] > priority:
            continue
        target[simplified_word] = pronunciations
        metadata[simplified_word] = (priority, False)


def _load_phrase_override_dict():
    signature = _build_phrase_override_signature()
    if os.path.exists(PHRASE_OVERRIDE_CACHE_PATH):
        try:
            with open(PHRASE_OVERRIDE_CACHE_PATH, "rb") as pickle_file:
                cache_payload = pickle.load(pickle_file)
            if cache_payload.get("signature") == signature:
                return cache_payload.get("data", {})
        except Exception:
            pass

    merged = {}
    metadata = {}
    for priority, file_path in enumerate([*EXTRA_PHRASE_OVERRIDE_PATHS, PHRASE_OVERRIDE_PATH]):
        if not os.path.exists(file_path):
            continue
        _merge_phrase_override_entries(merged, metadata, _read_phrase_override_file(file_path), priority)

    with open(PHRASE_OVERRIDE_CACHE_PATH, "wb") as pickle_file:
        pickle.dump({"signature": signature, "data": merged}, pickle_file)
    return merged


def read_dict():
    polyphonic_dict = {}
    polyphonic_dict.update(_read_rep_file(PP_DICT_PATH))
    polyphonic_dict.update(_read_rep_file(PP_FIX_DICT_PATH))
    return polyphonic_dict


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            polyphonic_dict = pickle.load(pickle_file)
    else:
        polyphonic_dict = read_dict()
        cache_dict(polyphonic_dict, CACHE_PATH)

    return polyphonic_dict


pp_dict = get_dict()
phrase_override_dict = _load_phrase_override_dict()


def get_phrase_pronunciation(word):
    value = phrase_override_dict.get(word, "")
    if value != "":
        return value
    value = pp_dict.get(word, "")
    if value != "" and len(word) > 1:
        return value
    return None


def correct_pronunciation(word, word_pinyins):
    local_override = get_phrase_pronunciation(word)
    if local_override is not None:
        return local_override
    new_pinyins = pp_dict.get(word, "")
    if new_pinyins == "":
        for idx, w in enumerate(word):
            w_pinyin = pp_dict.get(w, "")
            if w_pinyin != "":
                word_pinyins[idx] = w_pinyin[0]
        return word_pinyins
    return new_pinyins
