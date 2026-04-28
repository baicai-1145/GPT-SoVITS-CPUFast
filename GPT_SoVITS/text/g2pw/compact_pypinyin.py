"""在 pypinyin 加载前拦截，用紧凑格式存储 phrases_dict，省 ~22 MB。"""
import json
import os
import sys
import types


class CompactPhrasesDict(dict):
    def __getitem__(self, key):
        return [[s for s in p.split(",")] for p in dict.__getitem__(self, key).split("|")]


def install():
    if "pypinyin.phrases_dict" in sys.modules:
        mod = sys.modules["pypinyin.phrases_dict"]
        if isinstance(getattr(mod, "phrases_dict", None), CompactPhrasesDict):
            return
    import importlib.util
    spec = importlib.util.find_spec("pypinyin.phrases_dict")
    if spec is None:
        return
    json_path = os.path.join(os.path.dirname(spec.origin), "phrases_dict.json")

    cd = CompactPhrasesDict()
    with open(json_path, encoding="utf-8") as f:
        for k, v in json.load(f).items():
            dict.__setitem__(cd, k, "|".join(",".join(syl) for syl in v))

    fake_mod = types.ModuleType("pypinyin.phrases_dict")
    fake_mod.phrases_dict = cd
    fake_mod._json_path = json_path
    sys.modules["pypinyin.phrases_dict"] = fake_mod

    # 如果 pypinyin.constants 已经 import 了，也要替换
    if "pypinyin.constants" in sys.modules:
        sys.modules["pypinyin.constants"].PHRASES_DICT = cd
