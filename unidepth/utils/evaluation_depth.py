"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""


from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from chamfer_distance import ChamferDistance

from unidepth.utils.constants import DEPTH_BINS
from unidepth.utils.flow import flow_warp

chamfer_cls = ChamferDistance()


def kl_div(gt, pred, eps: float = 1e-6):
    depth_bins = DEPTH_BINS.to(gt.device)
    gt, pred = torch.bucketize(gt, boundaries=depth_bins, out_int32=True), torch.bucketize(pred, boundaries=depth_bins, out_int32=True)
    gt = torch.bincount(gt, minlength=len(depth_bins) + 1)
    pred = torch.bincount(pred, minlength=len(depth_bins) + 1)
    gt = gt / gt.sum()
    pred = pred / pred.sum()
    return  torch.sum(gt * (torch.log(gt + eps) - torch.log(pred + eps)) )


def chamfer_dist(tensor1, tensor2):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1,
        tensor2,
        x_lengths=x_lengths,
        y_lengths=y_lengths
    )
    return (torch.sqrt(dist1) + torch.sqrt(dist2)) / 2


def auc(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1,
        tensor2,
        x_lengths=x_lengths,
        y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    auc_value = torch.trapz(torch.tensor(precisions, device=tensor1.device), torch.tensor(recalls, device=tensor1.device))
    return auc_value


def delta(tensor1, tensor2, exponent):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < 1.25 ** exponent).to(torch.float32).mean()


def ssi(tensor1, tensor2, qtl=0.05):
    stability_mat = 1e-9 * torch.eye(2, device=tensor1.device)
    error = (tensor1 - tensor2).abs()
    mask = error < torch.quantile(error, 1-qtl)
    tensor1_mask = tensor1[mask]
    tensor2_mask = tensor2[mask]
    tensor2_one = torch.stack([tensor2_mask.detach(), torch.ones_like(tensor2_mask).detach()], dim=1)
    scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (tensor2_one.T @ tensor1_mask.unsqueeze(1))
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return tensor2 * scale + shift

    # tensor2_one = torch.stack([tensor2.detach(), torch.ones_like(tensor2).detach()], dim=1)
    # scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (tensor2_one.T @ tensor1.unsqueeze(1))
    # scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    # return tensor2 * scale + shift

def d1_ssi(tensor1, tensor2):
    delta_ = delta(tensor1, ssi(tensor1, tensor2), 1.0)
    return delta_


def d_auc(tensor1, tensor2):
    exponents = torch.linspace(0.01, 5.0, steps=100, device=tensor1.device)
    deltas = [delta(tensor1, tensor2, exponent) for exponent in exponents]
    return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0


def f1_score(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1,
        tensor2,
        x_lengths=x_lengths,
        y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    precisions = torch.tensor(precisions, device=tensor1.device)
    recalls = torch.tensor(recalls, device=tensor1.device)
    f1_thresholds = 2 * precisions * recalls / (precisions + recalls)
    f1_thresholds = torch.where(torch.isnan(f1_thresholds), torch.zeros_like(f1_thresholds), f1_thresholds)
    f1_value = torch.trapz(f1_thresholds) / len(thresholds)
    return f1_value


DICT_METRICS = {
    "d1": partial(delta, exponent=1.0),
    "d2": partial(delta, exponent=2.0),
    "d3": partial(delta, exponent=3.0),
    "rmse": lambda gt, pred: torch.sqrt(((gt - pred) ** 2).mean()),
    "rmselog": lambda gt, pred: torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean()),
    "arel": lambda gt, pred: (torch.abs(gt - pred) / gt).mean(),
    "sqrel": lambda gt, pred: (((gt - pred) ** 2) / gt).mean(),
    "log10": lambda gt, pred: torch.abs(torch.log10(pred) - torch.log10(gt)).mean(),
    "silog": lambda gt, pred: 100 * torch.std(torch.log(pred) - torch.log(gt)).mean(),
    "medianlog": lambda gt, pred: 100 * (torch.log(pred) - torch.log(gt)).median().abs(),
    "d_auc": d_auc,
    "d1_ssi": d1_ssi,
}


DICT_METRICS_3D = {
    "chamfer": lambda gt, pred, thresholds: chamfer_dist(gt.unsqueeze(0).permute(0, 2, 1), pred.unsqueeze(0).permute(0, 2, 1)),
    "F1": lambda gt, pred, thresholds: f1_score(gt.unsqueeze(0).permute(0, 2, 1), pred.unsqueeze(0).permute(0, 2, 1), thresholds=thresholds),
}


DICT_METRICS_FLOW = {
    "epe": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)),
    "epe1": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)) < 1,
    "epe3": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)) < 3,
    "epe5": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)) < 5,
}

DICT_METRICS_D = {
    "a1": lambda gt, pred: (torch.maximum((gt / pred), (pred / gt)) > 1.25 ** 1.0).to(torch.float32),
    "abs_rel": lambda gt, pred: (torch.abs(gt - pred) / gt),
}


def eval_depth(gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, max_depth=None):
    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if max_depth is not None:
            mask = torch.logical_and(mask, gt <= max_depth)
        for name, fn in DICT_METRICS.items():
            summary_metrics[name].append(fn(gt[mask], pred[mask]).mean())
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def eval_3d(gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, thresholds=None):
    summary_metrics = defaultdict(list)
    w_max = min(gts.shape[-1] // 4, 400)
    gts = F.interpolate(gts, (int(w_max * gts.shape[-2] / gts.shape[-1]), w_max), mode="nearest")
    preds = F.interpolate(preds, gts.shape[-2:], mode="nearest")
    masks = F.interpolate(masks.to(torch.float32), gts.shape[-2:], mode="nearest").bool()
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if not torch.any(mask):
            continue
        for name, fn in DICT_METRICS_3D.items():
            summary_metrics[name].append(fn(gt[:, mask.squeeze()], pred[:, mask.squeeze()], thresholds).mean())
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def rescale_flow(flow):
    _, h, w = flow.shape
    flow[0] = (flow[0] + 1) / 2 * (w - 1)
    flow[1] = (flow[1] + 1) / 2 * (h - 1)
    return flow


def eval_flow(gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor):
    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        for name, fn in DICT_METRICS_FLOW.items():
            mask = torch.logical_and(mask, torch.logical_and(gt > -1, gt < 1))
            gt = rescale_flow(gt)
            pred = rescale_flow(pred)
            summary_metrics[name].append(fn(gt[:, mask.squeeze()], pred[:, mask.squeeze()]).mean())
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def eval_pairwise(gts: dict[str, torch.Tensor], preds: torch.Tensor, masks: dict[str, torch.Tensor]):
    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts["depth"].shape[-2:], mode="bilinear")
    for i, (rgb_t, rgb_tp1, gt_depth_t, gt_depth_tp1, gt_flow_t, mask_flow_t, pred_t, pred_tp1, mask_t, mask_tp1, K) in enumerate(
        zip(gts["image"][:-1], gts["image"][1:], gts["depth"][-1:], gts["depth"][1:], gts["flow_fwd"], masks["flow_fwd_mask"], preds[:-1], preds[1:], masks["depth"][:-1], masks["depth"][1:], gts["K"])
    ):
        mask_tp1_warp, mask_w1 = flow_warp(mask_tp1, gt_flow_t)
        pred_tp1_warp, mask_w2 = flow_warp(pred_tp1, gt_flow_t)
        # rgb_tp1_warp, mask_w3 = flow_warp(rgb_tp1, gt_flow_t)
        gt_depth_tp1_warp, mask_w3 = flow_warp(gt_depth_tp1, gt_flow_t)
        mask_pred = mask_w2 & (pred_tp1_warp > 0.001) & mask_t & mask_tp1_warp & mask_flow_t
        mask_gt = mask_w2 & (gt_depth_tp1_warp > 0.001) & mask_t & mask_tp1_warp & mask_flow_t

        # mask_smooth = torch.exp(- 50 * torch.norm((rgb_t - rgb_tp1_warp).abs(), dim=0, keepdim=True))

        # from .visualization import colorize
        # pred_t[~mask] = 0.0
        # pred_tp1[~mask] = 0.0
        # pred_tp1_warp[~mask] = 0.0
        # pred_tp1_warp[mask] = ssi(pred_t[mask], pred_tp1_warp[mask])
        # Image.fromarray(colorize(gt_depth_t.squeeze().cpu().numpy(), vmin=pred_t[pred_t > 0.0].min().cpu().item())).save('pred_gt_t.png')
        # Image.fromarray(colorize(pred_t.squeeze().cpu().numpy(), vmin=pred_t[pred_t > 0.0].min().cpu().item())).save('pred_t.png')
        # Image.fromarray(colorize(pred_tp1.squeeze().cpu().numpy(), vmin=pred_t[pred_tp1 > 0.0].min().cpu().item())).save('pred_tp1.png')
        # Image.fromarray(colorize(pred_tp1_warp.squeeze().cpu().numpy(), vmin=pred_t[pred_tp1_warp > 0.0].min().cpu().item())).save('pred_tp1_warp.png')
        # Image.fromarray(colorize((gt_depth_t - pred_tp1_warp).abs().cpu().squeeze().numpy())).save('error.png')
        # rgb_t = ((rgb_t + 1) * 127.5).clip(0, 255).byte().cpu().permute(1,2,0).numpy()
        # rgb_tp1_warp = ((rgb_tp1_warp + 1) * 127.5).clip(0, 255).byte().cpu().permute(1,2,0).numpy()
        # Image.fromarray(rgb_t).save('rgb_t.png')
        # Image.fromarray(rgb_tp1_warp).save('rgb_tp1_warp.png')
        # Image.fromarray(mask.cpu().bool().squeeze().numpy()).save('pred_mask.png')
        # print((mask_smooth * (gt_depth_t - pred_tp1_warp)).abs().mean(), (gt_depth_t - pred_tp1_warp).abs().max(), (gt_depth_t - pred_tp1_warp).abs().min())
        # exit(1)
        # pred_t = ssi(gt_depth_t[mask], pred_t[mask]).clamp(1e-2)
        # pred_tp1_warp = ssi(gt_depth_t[mask], pred_tp1_warp[mask])
        # pred_tp1_warp = pred_tp1_warp[mask]

        # pred_tp1_warp = pred_tp1_warp[mask] * pred_t[mask].median() / pred_tp1_warp[mask].median()


        summary_metrics["OPW"].append(((pred_t[mask_pred] - pred_tp1_warp[mask_pred])).abs().mean())
        summary_metrics["OPW1"].append(((gt_depth_t[mask_gt] - gt_depth_tp1_warp[mask_gt])).abs().mean())
        # print("OPW", ((1 / pred_t[mask_pred] - 1 / pred_tp1_warp[mask_pred])).abs().mean())
        # print("OPW1", ((1 / gt_depth_t[mask_gt] - 1 / gt_depth_tp1_warp[mask_gt])).abs().mean())
        # from .geometric import unproject_points
        # pred_tp1_warp[mask] = ssi(pred_t[mask], pred_tp1_warp[mask])

        pred_t[mask_pred] = ssi(gt_depth_t[mask_pred], pred_t[mask_pred])
        pred_tp1_warp[mask_pred] = ssi(gt_depth_t[mask_pred], pred_tp1_warp[mask_pred])
        # pred_t_3d = unproject_points(pred_t.unsqueeze(0), K.unsqueeze(0)).squeeze()
        # gt_depth_t_3d = unproject_points(gt_depth_t.unsqueeze(0), K.unsqueeze(0)).squeeze()
        # pred_tp1_warp_3d = unproject_points(pred_tp1_warp.unsqueeze(0), K.unsqueeze(0)).squeeze()
        # gt_depth_tp1_warp_3d = unproject_points(gt_depth_tp1_warp.unsqueeze(0), K.unsqueeze(0)).squeeze()

        # pred_scene_flow = (pred_t_3d - pred_tp1_warp_3d)
        # gt_scene_flow = (gt_depth_t_3d - gt_depth_tp1_warp_3d)
        # error_scene_flow = torch.norm(gt_scene_flow - pred_scene_flow, dim=0, keepdim=True)[mask]
        # gt_scene_flow_norm = torch.norm(gt_scene_flow, dim=0, keepdim=True)[mask]
        # pred_scene_flow_norm = torch.norm(pred_scene_flow, dim=0, keepdim=True)[mask]
        # print("GT", gt_scene_flow_norm.min(), gt_scene_flow_norm.max(), gt_scene_flow_norm.mean(), (gt_scene_flow_norm == 0.0).float().mean())
        # print("PR", pred_scene_flow_norm.min(), pred_scene_flow_norm.max(), pred_scene_flow_norm.mean(), (pred_scene_flow_norm == 0.0).float().mean())
        # summary_metrics["DIF"].append((error_scene_flow / gt_scene_flow_norm.clip(min=1e-2)).mean())

        pred_t = pred_t[mask_pred]
        pred_tp1_warp = pred_tp1_warp[mask_pred]
        summary_metrics["aTC"].append(((pred_t - pred_tp1_warp) / pred_t).abs().mean())
        summary_metrics["rTC"].append(delta(pred_t, pred_tp1_warp, 1.0))
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def compute_aucs(gt, pred, mask, uncertainties, steps=50, metrics=["abs_rel"]):
    dict_ = {}
    x_axis = torch.linspace(0, 1, steps=steps+1, device=gt.device)
    quantiles = torch.linspace(0, 1 - 1/steps, steps=steps, device=gt.device)
    zer = torch.tensor(0.0, device=gt.device)
    # revert order (high uncertainty first)
    uncertainties = - uncertainties[mask]
    gt = gt[mask]
    pred = pred[mask]
    true_uncert = {metric: - DICT_METRICS_D[metric](gt, pred) for metric in metrics}
    # get percentiles for sampling and corresponding subsets
    thresholds = torch.quantile(uncertainties, quantiles)
    subs = [(uncertainties >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    for metric in metrics:
        opt_thresholds = torch.quantile(true_uncert[metric], quantiles)
        opt_subs = [(true_uncert[metric] >= t) for t in opt_thresholds]
        sparse_curve = torch.stack([DICT_METRICS[metric](gt[sub], pred[sub]) for sub in subs] + [zer], dim=0)
        opt_curve = torch.stack([DICT_METRICS[metric](gt[sub], pred[sub]) for sub in opt_subs] + [zer], dim=0)
        rnd_curve =  DICT_METRICS[metric](gt, pred)

        dict_[f"AUSE_{metric}"] = torch.trapz(sparse_curve - opt_curve, x=x_axis)
        dict_[f"AURG_{metric}"] = rnd_curve - torch.trapz(sparse_curve, x=x_axis)

    return dict_


def eval_depth_uncertainties(gts: torch.Tensor, preds: torch.Tensor, uncertainties:torch.Tensor, masks: torch.Tensor, max_depth=None):
    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask, uncertainty) in enumerate(zip(gts, preds, masks, uncertainties)):
        if max_depth is not None:
            mask = torch.logical_and(mask, gt < max_depth)
        for name, fn in DICT_METRICS.items():
            summary_metrics[name].append(fn(gt[mask], pred[mask]))
        for name, val in compute_aucs(gt, pred, mask, uncertainty).items():
            summary_metrics[name].append(val)
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}

