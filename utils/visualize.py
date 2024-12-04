# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import numpy as np
import torch
import cv2
# utils
from .metric import parse_single_lane_sequence
from utils.llamas_utils import get_dcolors

def image_torch_to_numpy(x: torch.Tensor):
    if x.size(0) == 3:
        x = x.permute(1, 2, 0) # [3, H, W] -> [H, W, 3]
    return x.detach().cpu().numpy()


def vis_lane_circle(
    image: Union[np.ndarray, torch.Tensor],
    lane: torch.Tensor,
) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image_torch_to_numpy(image)
    image = cv2.resize(image, dsize=(1276, 717))
    img_size = image.shape[:2]
    # parse lanes
    lanes = parse_single_lane_sequence(lane, img_size=img_size)
    # gather all markers
    colors = get_dcolors(len(lanes))
    # draw markers
    for i, lane in enumerate(lanes):
        for x, y in lane:
            cv2.circle(image, (x, y), radius=6, color=colors[i], thickness=-1)
    return image


def vis_lane_line(
    image: Union[np.ndarray, torch.Tensor],
    lane: Union[List[int], torch.Tensor],
) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image_torch_to_numpy(image)
    image = cv2.resize(image, dsize=(1276, 717))
    img_size = image.shape[:2]
    # parse lanes
    lanes = parse_single_lane_sequence(lane, img_size=img_size)
    # gather all colors
    colors = get_dcolors(len(lanes))
    # draw lines
    for i, lane in enumerate(lanes):
        for p1, p2 in zip(lane[:-1], lane[1:]):
            cv2.line(image, p1, p2, color=colors[i], thickness=10)
    return image