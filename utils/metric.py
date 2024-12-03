# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import cv2
import torch
import numpy as np
# utils
from .llamas_utils import interpolate_lane, culane_metric, remove_consecutive_duplicates
from dataset.const import *


def dequantize_point(x: int, bins: int = 1000) -> float:
    return (x - 1) / (bins - 1)


def parse_single_lane_sequence(x: torch.Tensor, img_size: Tuple[int, int]) -> List[List[Tuple[float, float]]]:
    """
    Parse a single lane sequence.
    Args:
        x (torch.Tensor): [seq]
    Returns:
        List[List[Tuple[float, float]]]: [[(x, y)]]
    """
    x = x.detach().cpu().tolist()
    H, W = img_size
    # parse lanes
    lanes = []
    new_lane = []
    for _, token in enumerate(x):
        if token == TOKEN_END: break
        if token in [TOKEN_START, TOKEN_ANCHOR, TOKEN_PAD]: continue
        
        if token == TOKEN_LANE:
            lanes.append(new_lane.copy())
            new_lane.clear()
        else:
            new_lane.append(token)
    # dequantize
    for lane in lanes:
        for i, point in enumerate(lane):
            lane[i] = dequantize_point(point)
    # to list((x, y))
    final_lanes = []
    for lane in lanes:
        points_lane = [(int(lane[i] * W), int(lane[i+1] * H)) 
                            for i in range(0, len(lane) - 1, 2)]
        # unique and sorted
        points_lane = remove_consecutive_duplicates(points_lane)
        if len(points_lane) > 1:
            final_lanes.append(points_lane.copy())
    return final_lanes


def evaluate_single_batch(
    x: torch.Tensor,        # [seq]
    gt: torch.Tensor,       # [seq]
    img_size: Tuple[int, int] = (717, 1276)
) -> Tuple[float, float, float]:
    # to [[(x, y), ...], ...]
    pred = parse_single_lane_sequence(x, img_size=img_size)
    gt = parse_single_lane_sequence(gt, img_size=img_size)

    # interpolate
    interp_pred = np.array([interpolate_lane(lane) for lane in pred])
    interp_gt = np.array([interpolate_lane(lane) for lane in gt])

    metric = culane_metric(interp_pred, interp_gt, img_shape=img_size)[0.5]
    # [tp, fp, fn]
    return metric

def f1_evaluate(
    x: torch.Tensor,    # [b, seq]
    gt: torch.Tensor,   # [b, seq]
    img_size: Tuple[int, int] = (717, 1276)
) -> float:
    """
    Evaluate the model.
    Args:
        x (torch.Tensor): predicted sequences [b, seq]
        gt (torch.Tensor): ground truth sequences [b, seq]
    Returns:
        float: metric
    """
    metrics = []
    for b in range(x.size(0)):
        metric = evaluate_single_batch(x[b], gt[b], img_size=img_size)
        metrics.append(metric)

    tp = sum([metric[0] for metric in metrics])
    fp = sum([metric[1] for metric in metrics])
    fn = sum([metric[2] for metric in metrics])

    precision = float(tp) / (tp + fp) if tp != 0 else 0
    recall = float(tp) / (tp + fn) if tp != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
