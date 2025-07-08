import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 有些层在本 repo 里做了 “打包线性层” 优化（把 q,k,v 合成一个大权重或 gate、up 合成一个权重）。
                # 逻辑：当权重名里出现 "q_proj" → 把 "q_proj" 替换成 "qkv_proj" 去模型里找对应参数；额外传入 shard_id="q" 表示它属于大矩阵里的 q 区块；weight_loader 决定怎么把这片张量 copy 到大矩阵里正确 offset。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else: # 正常的权重名字，直接copy参数
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
