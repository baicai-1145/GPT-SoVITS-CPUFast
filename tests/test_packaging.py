from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from gpt_sovits_cpufast.runtime import PACKAGE_ROOT, configure_runtime


class PackagingSmokeTests(unittest.TestCase):
    def test_console_script_targets_are_packaged(self) -> None:
        for script in (
            "gpt_sovits_cpufast/webui.py",
            "gpt_sovits_cpufast/api.py",
            "gpt_sovits_cpufast/api_v2.py",
            "GPT_SoVITS/inference_cli.py",
        ):
            self.assertTrue((PACKAGE_ROOT / script).is_file(), script)

    def test_runtime_roots_are_configured_for_external_data_dir(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        temp_root = project_root / ".tmp"
        temp_root.mkdir(exist_ok=True)
        old_cwd = Path.cwd()
        old_data = os.environ.pop("GPT_SOVITS_DATA_DIR", None)
        old_code = os.environ.pop("GPT_SOVITS_CODE_DIR", None)
        old_config = os.environ.pop("GPT_SOVITS_TTS_CONFIG", None)
        try:
            with tempfile.TemporaryDirectory(dir=temp_root) as directory:
                os.environ["GPT_SOVITS_DATA_DIR"] = directory
                data_dir = configure_runtime()
                self.assertEqual(data_dir, Path(directory).resolve())
                self.assertEqual(Path.cwd(), Path(directory).resolve())
                self.assertEqual(Path(os.environ["GPT_SOVITS_CODE_DIR"]), PACKAGE_ROOT)
                self.assertEqual(
                    Path(os.environ["GPT_SOVITS_TTS_CONFIG"]),
                    PACKAGE_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml",
                )
                self.assertIn(str(PACKAGE_ROOT / "GPT_SoVITS"), sys.path)
        finally:
            os.chdir(old_cwd)
            for name, value in (
                ("GPT_SOVITS_DATA_DIR", old_data),
                ("GPT_SOVITS_CODE_DIR", old_code),
                ("GPT_SOVITS_TTS_CONFIG", old_config),
            ):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
