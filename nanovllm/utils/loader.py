import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, checkpoint: str | None = None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    def update_model(weight_name, weight):
        if weight_name.startswith("output."):
            weight_name = weight_name.replace("output.", "lm_head.")
        elif weight_name.startswith("tok_embeddings"):
            weight_name = weight_name.replace("tok_embeddings", "model.embed_tokens")
        elif not weight_name.startswith("model.") and not weight_name.startswith("lm_head."):
            weight_name = "model." + weight_name
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                if shard_id is not None:
                    weight_loader(param, weight, shard_id)
                else:
                    weight_loader(param, weight)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)

    if checkpoint is not None:
        weight = torch.load(os.path.join(checkpoint, "model_state_rank_0.pth"), weights_only=True)
        for weight_name in weight:
            update_model(weight_name, weight[weight_name])
    else:
        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    update_model(weight_name, f.get_tensor(weight_name))
