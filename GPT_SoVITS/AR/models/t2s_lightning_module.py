import os
import sys

import torch
from torch import nn

now_dir = os.getcwd()
sys.path.append(now_dir)

from AR.models.t2s_model import Text2SemanticDecoder


class Text2SemanticLightningModule(nn.Module):
    def __init__(
        self,
        config,
        output_dir,
        is_train=True,
        build_t2s_transformer: bool = True,
        build_h_module: bool = True,
    ):
        super().__init__()
        if is_train:
            raise RuntimeError("Training code has been removed from this inference-only build.")
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(
            config=config,
            top_k=self.top_k,
            build_t2s_transformer=build_t2s_transformer,
            build_h_module=build_h_module,
        )

    def training_step(self, *args, **kwargs):
        raise RuntimeError("Training code has been removed from this inference-only build.")

    def validation_step(self, *args, **kwargs):
        raise RuntimeError("Training code has been removed from this inference-only build.")

    def configure_optimizers(self):
        raise RuntimeError("Training code has been removed from this inference-only build.")

    def load_state_dict(self, state_dict, strict: bool = True, rebuild_transformer: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        if rebuild_transformer:
            self.model.rebuild_t2s_transformer()
        return result

    def load_inference_only_state_dict(self, state_dict, strict: bool = True):
        model_state_dict = {}
        model_prefix = "model."
        h_prefix = "model.h."
        for key, value in state_dict.items():
            if not key.startswith(model_prefix):
                continue
            if key.startswith(h_prefix):
                continue
            model_state_dict[key[len(model_prefix) :]] = value
        result = self.model.load_state_dict(model_state_dict, strict=strict)
        self.model.rebuild_t2s_transformer_from_state_dict(state_dict, prefix=model_prefix)
        return result
