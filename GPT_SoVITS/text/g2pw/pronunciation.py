import ast
import os
import pickle

current_file_path = os.path.dirname(__file__)
CACHE_PATH = os.path.join(current_file_path, "polyphonic.pickle")
PP_DICT_PATH = os.path.join(current_file_path, "polyphonic.rep")
PP_FIX_DICT_PATH = os.path.join(current_file_path, "polyphonic-fix.rep")
PHRASE_OVERRIDE_PATH = os.path.join(current_file_path, "phrase_overrides.rep")


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
phrase_override_dict = _read_rep_file(PHRASE_OVERRIDE_PATH) if os.path.exists(PHRASE_OVERRIDE_PATH) else {}


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
