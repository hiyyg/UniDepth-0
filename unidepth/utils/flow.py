"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""


import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]
    stacks = [x, y]
    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]
    if device is not None:
        grid = grid.to(device)

    return grid

def normalize_coords(coords, h, w):
    c = torch.tensor([(w - 1) / 2., (h - 1) / 2.], device=coords.device).view(1, 2, 1, 1)
    return (coords - c) / c


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros'):
    if sample_coords.shape[1] == 2:
        sample_coords = sample_coords.permute(0, 2, 3, 1)
    img = F.grid_sample(img, sample_coords, mode=mode, padding_mode=padding_mode, align_corners=True)
    mask = torch.logical_or(sample_coords >= -1, sample_coords <= 1).permute(0, 3, 1, 2)  # [B, 2, H, W]
    return img, mask[:, :1] & mask[:, 1:]


def flow_warp(feature, flow, padding_mode='zeros'):
    dtype = feature.dtype
    feature_shape = feature.shape
    flow_shape = flow.shape
    leading_ndim = max(0, (4 - len(feature_shape)))
    feature = feature.reshape((1,) * leading_ndim + feature_shape)
    flow = flow.reshape((1,) * leading_ndim + flow_shape)
    b, c, h, w = feature.shape
    assert flow.shape[1] == 2
    grid = coords_grid(b, h, w).to(flow.device) 
    grid = normalize_coords(grid, h, w) + flow  # [B, 2, H, W]
    warped_feature, mask = bilinear_sample(feature.float(), grid, padding_mode=padding_mode)
    return warped_feature.reshape(feature_shape).to(dtype), mask.squeeze(0) # FIXME, should be featue dim with 1 set to 1
