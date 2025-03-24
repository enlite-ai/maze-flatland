"""File holding the runner for XGBoost model."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from maze.core.annotations import override
from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset
from maze.core.utils.config_utils import SwitchWorkingDirectoryToInput
from maze.core.utils.factory import Factory
from maze.train.trainers.common.evaluators.evaluator import Evaluator
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.utils.get_size_of_objects import getsize
from maze_flatland.agents.xgboost_policy import XGBoostModel
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score


class DatasetSplitMode(IntEnum):
    flat = 0
    trajectory_based = 1


def sort_and_split(data: list[any], pivot: int, idxs: list[int]) -> tuple[list[any], list[any]]:
    """Given an array returns 2 arrays split at the pivot position.
    :param data: The array to be split.
    :param pivot: The pivot position.
    :param idxs: The indices defining the sorting of the array."""
    _data = np.array(data)[idxs]
    return _data[:pivot], _data[pivot:]


def create_training_data(
    dataset: InMemoryDataset, split_mode: DatasetSplitMode, split_ratio: int | float
) -> tuple[tuple[list, list, list], tuple[list, list, list]]:
    """Split a dataset into train and validation.
    :param dataset: The dataset to be split.
    :param split_mode: The split mode.
    :param split_ratio: ratio of validation set.
    :return: a pair with a triplet each containing the observation, actions and masks.
    """
    labels = []
    inputs = []
    masks = []
    assert 0 <= split_ratio < 100
    if split_mode == DatasetSplitMode.flat:
        for sr in dataset.step_records:
            for ssr in sr.substep_records:
                labels.append(list(ssr.action.values())[0])
                assert len(ssr.observation) <= 2, 'Not supported.'
                for k, v in ssr.observation.items():
                    if k.endswith('_mask'):
                        masks.append(v)
                    else:
                        inputs.append(v)

    else:
        raise NotImplementedError  # currently not supported the split by trajectory id.
    assert len(labels) == len(inputs) and len(masks) in [0, len(labels)]
    if split_ratio == 0:
        return (inputs, labels, masks), ([], [], [])
    indices = list(range(len(inputs)))
    split_idx = int((1 - (split_ratio / 100)) * len(inputs))
    random.shuffle(indices)
    x_train, x_test = sort_and_split(inputs, split_idx, indices)
    y_train, y_test = sort_and_split(labels, split_idx, indices)
    if len(masks) > 0:
        masks_train, masks_test = sort_and_split(masks, split_idx, indices)
    else:
        masks_train = []
        masks_test = []
    return (x_train, y_train, masks_train), (x_test, y_test, masks_test)


@dataclass
class XGBoostRunner(TrainingRunner):
    """Dev runner for XGBoost model."""

    dataset: DictConfig

    @staticmethod
    def _clean_policy_data():
        policy_files = ['critic_', 'spaces_config.pkl', 'policy_']
        for filename in os.listdir():
            if any(filename.startswith(f) for f in policy_files):
                os.remove(filename)

    # pylint: disable= attribute-defined-outside-init
    def setup(self, cfg: DictConfig):
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.setup`.
        """
        super().setup(cfg)
        self._clean_policy_data()
        # load dataset
        with SwitchWorkingDirectoryToInput(cfg.input_dir):
            dataset = Factory(base_type=InMemoryDataset).instantiate(self.dataset, conversion_env_factory=None)
        size_in_byte, size_in_gbyte = getsize(dataset)
        print(f'Size of loaded dataset: {size_in_byte} -> {round(size_in_gbyte, 3)} GB')
        # split dataset
        random.seed(self.maze_seeding.generate_env_instance_seed())
        self.train_data, self.test_data = create_training_data(
            dataset, DatasetSplitMode.flat, cfg.algorithm.validation_percentage
        )
        print(f'Sample Counts (train: {len(self.train_data[0])}, validation: {len(self.test_data[0])})')
        assert len(self.train_data[0]) != 0 and len(self.test_data[0]) >= 0
        assert 'gamma' in cfg.algorithm
        assert 'reg_alpha' in cfg.algorithm
        assert 'reg_lambda' in cfg.algorithm

        gamma = cfg.algorithm.gamma
        reg_alpha = cfg.algorithm.reg_alpha
        reg_lambda = cfg.algorithm.reg_lambda

        # create xgboost
        self.policy = XGBoostModel(
            n_estimators=cfg.algorithm.n_estimators,
            max_depth=cfg.algorithm.max_depth,
            lr=cfg.algorithm.learning_rate,
            objective=cfg.algorithm.objective,
            seed=self.maze_seeding.generate_env_instance_seed(),
            n_processes=cfg.algorithm.n_processes,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

    @override(TrainingRunner)
    def run(
        self,
        n_epochs: Optional[int] = None,
        evaluator: Optional[Evaluator] = None,
        eval_every_k_iterations: Optional[int] = None,
    ) -> None:
        """
        Run the training master node.
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.run`.
        :param evaluator: not used.
        :param n_epochs: not used
        :param eval_every_k_iterations: not used.
        """
        _ = (n_epochs, evaluator, eval_every_k_iterations)
        tstart = time.time()
        x_train, y_train, _ = self.train_data
        x_test, y_test, _ = self.test_data
        _ = self.policy.model.fit(x_train, y_train)
        print(f'Training time: {time.time() - tstart:.2f}s')
        if len(x_test) == 0 and len(y_test) == 0:
            print('Validation not possible.')
        else:
            # test the accuracy
            y_pred = self.policy.predict(x_test)
            accuracy = accuracy_score(y_pred, y_test)
            print(f'Accuracy of trained model: {accuracy}')
        # save model.
        self.policy.save_model(os.path.abspath('./'))
