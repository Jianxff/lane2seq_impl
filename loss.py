# standard library
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
# third party
import cv2
import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super(WeightedCrossEntropyLoss, self).__init__()
        # weight for cross entropy loss
        weight = torch.ones(1000 + 7)
        # 0 for format transcription
        weight[1004] = 0
        weight[1005] = 0
        weight[1006] = 0
        
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=0,
        )

    def forward(
        self,
        x: torch.Tensor,        # [b, seq, n_bins + 7]
        target: torch.Tensor,   # [b, seq]
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): predicted logits [b, seq, n_bins + 7]
            target (torch.Tensor): target sequence [b, seq]
        Returns:
            torch.Tensor: loss
        """
        # flatten
        x = x.view(-1, x.size(-1))
        target = target.view(-1)

        return self.loss(x, target)
    
