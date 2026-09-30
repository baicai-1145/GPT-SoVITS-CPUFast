"""GPT-SoVITS inference implementation.

The implementation uses short absolute imports such as ``text`` and ``module``.
Keep this package directory on ``sys.path`` so those imports resolve to this
installed distribution rather than to unrelated global packages.
"""

from pathlib import Path
import sys

_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)
