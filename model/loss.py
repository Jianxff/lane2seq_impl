# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import cv2
import torch
import torch.nn as nn
from dataset.const import SPECIAL_CODE_NUM

class CrossEntropy:
    def __init__(self, reduction = 'mean', device: str = 'cuda') -> None:
        super(CrossEntropy, self).__init__()
        # # weight for cross entropy loss
        # weight = torch.ones(1000 + SPECIAL_CODE_NUM)
        # # 0 for format transcription
        # weight[1004] = 0
        # weight[1005] = 0
        # weight[1006] = 0
        
        self.loss = nn.CrossEntropyLoss(
            ignore_index=0,
            reduction=reduction,
        ).to(device)

    def compute(
        self,
        x: torch.Tensor,        # [b, seq, n_bins + k]
        target: torch.Tensor,   # [b, seq]
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): predicted logits [b, seq, n_bins + k]
            target (torch.Tensor): target sequence [b, seq]
        Returns:
            torch.Tensor: loss
        """
        # flatten
        x = x.view(-1, x.size(-1))
        target = target.view(-1)

        return self.loss(x, target)
    
