import torch
import numpy as np
from PIL import Image


def demo(model):
    rgb_torch = torch.from_numpy(np.array(Image.open("assets/demo/rgb.rgb"))).permute(2, 0, 1)
    intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))

    # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
    predictions = model.infer(rgb_torch, intrinsics_torch)
    
    print("Available predictions: ", predictions.keys())
    depth_predictions = predictions["depth"].squeeze().cpu().numpy()
    depth_gt = np.array(Image.open("assets/demo/depth.png")).astype(float) / 1000.0


if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    model = torch.hub.load("lpiccinelli-eth/unidepth", "UniDepthV1_ViTL14", pretrained=True, trust_repo=True)
    demo(model)