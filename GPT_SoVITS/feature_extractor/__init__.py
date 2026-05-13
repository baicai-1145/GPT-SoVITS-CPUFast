import importlib


class _LazyContentModuleMap(dict):
    def __missing__(self, key):
        module_name = {"cnhubert": "cnhubert", "whisper": "whisper_enc"}[key]
        module = importlib.import_module(f"{__name__}.{module_name}")
        self[key] = module
        return module


content_module_map = _LazyContentModuleMap()
