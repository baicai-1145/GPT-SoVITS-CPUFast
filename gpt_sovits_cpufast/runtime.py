"""Runtime setup shared by source-tree and installed launchers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def configure_runtime() -> Path:
    """Configure code/data roots and return the writable data directory."""
    data_dir = Path(os.environ.get("GPT_SOVITS_DATA_DIR", Path.cwd())).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GPT_SOVITS_DATA_DIR"] = str(data_dir)

    os.environ.setdefault("GPT_SOVITS_CODE_DIR", str(PACKAGE_ROOT))
    os.environ.setdefault(
        "GPT_SOVITS_TTS_CONFIG",
        str(PACKAGE_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml"),
    )

    for import_root in (PACKAGE_ROOT, PACKAGE_ROOT / "GPT_SoVITS"):
        import_root = str(import_root)
        if import_root not in sys.path:
            sys.path.insert(0, import_root)

    os.chdir(data_dir)
    return data_dir
