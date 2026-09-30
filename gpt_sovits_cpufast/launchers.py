"""Console entry points for running the inference scripts after installation."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from typing import NoReturn

from .runtime import configure_runtime


def _runtime_dir() -> Path:
    return configure_runtime()


def _run(module: str) -> NoReturn:
    _runtime_dir()
    module_spec = importlib.util.find_spec(module)
    if module_spec is None or not module_spec.origin:
        raise RuntimeError(f"Installed module is missing: {module}")
    sys.argv[0] = module_spec.origin
    runpy.run_module(module, run_name="__main__")
    raise SystemExit(0)


def webui() -> NoReturn:
    _run("gpt_sovits_cpufast.webui")


def api() -> NoReturn:
    _run("gpt_sovits_cpufast.api")


def api_v2() -> NoReturn:
    _run("gpt_sovits_cpufast.api_v2")


def cli() -> NoReturn:
    _run("GPT_SoVITS.inference_cli")
