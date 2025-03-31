"""File holding the custom loss definition for BC cloning."""
from __future__ import annotations

from typing import Union

import gymnasium as gym
import torch
from maze.train.trainers.imitation.bc_loss import BCLoss
from torch import Tensor


class ClippedLoss(torch.nn.Module):
    """Overrides a given torch loss function to clip its value within a specified range.
    :param original_loss: the loss function to clip.
    :param min_value: the minimum value to clip the loss to.
    :param max_value: the maximum value to clip the loss to.
    """

    def __init__(self, original_loss: torch.nn.Module, min_value: float, max_value: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert min_value < max_value
        self.min_clip = min_value
        self.max_clip = max_value
        self.loss = original_loss

    def forward(self, _input: Tensor, target: Tensor) -> Tensor:
        original_loss = self.loss(_input, target)
        clipped_loss = torch.clip(original_loss, self.min_clip, self.max_clip)
        return clipped_loss


class ClippedBCLoss(BCLoss):
    """Overrides the traditional BCLoss to clip the discrete loss to a specified range.
        see ~maze.train.trainers.imitation.bc_loss.BCLoss
    :param min_clip: minimum value to clip the loss to.
    :param max_clip: maximum value to clip the loss to.
    """

    def __init__(
        self,
        action_spaces_dict: dict[Union[int, str], gym.spaces.Dict],
        entropy_coef: float,
        min_clip: float,
        max_clip: float,
    ):
        super().__init__(action_spaces_dict=action_spaces_dict, entropy_coef=entropy_coef)
        # override loss discrete with the clipped one.
        self.loss_discrete = ClippedLoss(self.loss_discrete, min_clip, max_clip)
