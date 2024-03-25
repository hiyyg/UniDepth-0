dependencies = ["torch"]

import os
import json

import torch
import huggingface_hub

from unidepth.models import UniDepthV1


def UniDepthV1_ViTL14(pretrained):
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", "config_v1_vitl14.json")) as f:
        config = json.load(f)
    
    model = UniDepthV1.build(config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id="lpiccinelli/UniDepth", filename="unidepth_v1_vitl14.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print("UniDepthV1_ViTL14 is loaded")
        print(f"missing keys: {info.missing_keys}\nadditional keys: {info.unexpected_keys}")

    return model


def UniDepthV1_ConvNextL(pretrained):
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", "config_v1_cnvnxtl.json")) as f:
        config = json.load(f)
    
    model = UniDepthV1.build(config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id="lpiccinelli/UniDepth", filename="unidepth_v1_cnvnxtl.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print("UniDepthV1_ConvNextL is loaded")
        print(f"missing keys: {info.missing_keys}\nadditional keys: {info.unexpected_keys}")

    return model