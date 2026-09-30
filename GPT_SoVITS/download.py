import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MODULE_DIR)
for path in (REPO_ROOT, MODULE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from gpt_sovits_cpufast.runtime import configure_runtime

configure_runtime()

from text.g2pw import G2PWPinyin

g2pw = G2PWPinyin(
    model_dir="GPT_SoVITS/text/G2PWModel",
    model_source="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    v_to_u=False,
    neutral_tone_with_five=True,
)
