import os
import sys

import torch
from torch import nn

now_dir = os.getcwd()
sys.path.append(now_dir)

from AR.models.t2s_model import Text2SemanticDecoder


class Text2SemanticLightningModule(nn.Module):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        if is_train:
            raise RuntimeError("Training code has been removed from this inference-only build.")
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)

    def training_step(self, *args, **kwargs):
        raise RuntimeError("Training code has been removed from this inference-only build.")

    def validation_step(self, *args, **kwargs):
        raise RuntimeError("Training code has been removed from this inference-only build.")

    def configure_optimizers(self):
        raise RuntimeError("Training code has been removed from this inference-only build.")

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self.model.rebuild_t2s_transformer()
        return result
