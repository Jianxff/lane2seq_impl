# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import cv2
import numpy as np
import torch
import torch.nn as nn
# utils
from utils.llamas_utils import interpolate_lane, culane_metric
from utils.metric import parse_single_lane_sequence
from utils.line_iou import line_iou
from utils.batch_util import batch_fn


class REINFORCE:
    def __init__(
        self,
        lambda_2: float = 0.3,
        lambda_5: float = 1.0,
        w_ce: float = 0.3,
        img_size: Tuple[int, int] = (717, 1276),
    ) -> None:
        
        self.lambda_2 = lambda_2
        self.lambda_5 = lambda_5
        self.w_ce = w_ce
        self.img_size = img_size


    def reward_single_batch(
        self,
        x: np.ndarray,        # [seq]
        interp_gt: np.ndarray
    ) -> np.ndarray:
        # parse to [[(x, y), ...], ...]
        pred = parse_single_lane_sequence(x, self.img_size)
        # interpolate
        interp_pred = np.array([interpolate_lane(lane) for lane in pred])

        reward = 0        
        # metric
        tp, fp, fn, tp_index = culane_metric(interp_pred, interp_gt, \
                                             img_shape=self.img_size)[0.5]
        if tp > 0:
            for (x, y) in zip(tp_index[0], tp_index[1]):
                tp_lane = interp_pred[x]
                gt_lane = interp_gt[y]
                tp_lane_x = tp_lane[:, 0]
                gt_lane_x = gt_lane[:, 0]
                # calculate reward
                liou = line_iou(tp_lane_x, gt_lane_x, img_w=self.img_size[1])
                # calculate matched Euclidean distance
                distance = np.linalg.norm(tp_lane - gt_lane, axis=1)
                distance = np.mean(distance)
                d_r = 1 - distance / self.img_size[0]
                reward += (liou + d_r)
            reward /= tp

        # FP
        fp_rate = (fp / (tp + fp)) if (tp + fp) > 0 else 1        
        reward -= (self.lambda_2 * fp_rate)
    
        return reward

    def reward(
        self,
        x: np.ndarray,        # [b, seq]
        gt: List[np.ndarray]
    ) -> torch.Tensor:
        # parallel using threading
        rewards = batch_fn(self.reward_single_batch, x.shape[0], x, gt)
        rewards = np.array(rewards, dtype=np.float32)
        return rewards


    @staticmethod
    def batch_sample(
        x: torch.Tensor,        # [b, seq, n_bins + k]
        num_samples: int = 1
    ) -> torch.Tensor:
        # size
        batch_size, seq_len, dims = x.size()
        x = torch.softmax(x, dim=-1)
        # probability sampling
        x = x.view(-1, dims) # [b * seq, n_bins + k]
        # sample
        samples = torch.multinomial(x, num_samples=num_samples) # [b * seq, num_samples]
        # dims recover
        samples = samples.view(batch_size, seq_len, num_samples) # [b, seq, num_samples]
        return samples
    

    def compute(
        self,
        x: torch.Tensor,        # [b, seq, n_bins + k]
        target: torch.Tensor,   # [b, seq]
    ) -> torch.Tensor:
        batch_size, seq_len, n_bins = x.shape
        x_softmax = torch.log_softmax(x, dim=-1)

        # sample
        samples = self.batch_sample(x, 2)
        y_sample = samples[:, :, 0]     # [b, seq]
        y_baseline = samples[:, :, 1]   # [b, seq]

        # interpolate ground truth
        def interpolate_single_batch(target):
            gt = parse_single_lane_sequence(target, self.img_size)
            interp_gt = np.array([interpolate_lane(lane) for lane in gt])
            return interp_gt
        lanes_gt = batch_fn(interpolate_single_batch, batch_size, target)

        # calculate reward
        reward_sample = self.reward(y_sample.cpu().numpy(), lanes_gt)
        reward_baseline = self.reward(y_baseline.cpu().numpy(), lanes_gt)
        r = reward_sample - reward_baseline
        r = torch.from_numpy(r).float().to(x.device)
        
        # calculate loss
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=n_bins).float() \
                        * r[:, None, None]
        loss_reward = - torch.sum(x_softmax * target_one_hot, dim=-1)
        loss_reward = loss_reward.mean()

        return loss_reward