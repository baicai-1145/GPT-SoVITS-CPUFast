import os
import pickle

current_file_path = os.path.dirname(__file__)
CACHE_PATH = os.path.join(current_file_path, "polyphonic.pickle")
PP_DICT_PATH = os.path.join(current_file_path, "polyphonic.rep")
PP_FIX_DICT_PATH = os.path.join(current_file_path, "polyphonic-fix.rep")


def cache_dict(polyphonic_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(polyphonic_dict, pickle_file)


def read_dict():
    polyphonic_dict = {}
    with open(PP_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
    with open(PP_FIX_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
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


def correct_pronunciation(word, word_pinyins):
    new_pinyins = pp_dict.get(word, "")
    if new_pinyins == "":
        for idx, w in enumerate(word):
            w_pinyin = pp_dict.get(w, "")
            if w_pinyin != "":
                word_pinyins[idx] = w_pinyin[0]
        return word_pinyins
    return new_pinyins
