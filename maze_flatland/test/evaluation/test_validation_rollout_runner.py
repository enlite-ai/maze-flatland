"""File holdings the tests for the validation rollout runner."""
from __future__ import annotations

from typing import Optional

import numpy as np
from hydra import compose, initialize_config_module
from maze_flatland.evaluation.validation_rollout_runner import ValidationRolloutRunner
from omegaconf import DictConfig


def read_hydra_config_with_overrides(config_module: str, config_name: str, overrides: list[str]) -> DictConfig:
    """Read and assemble a hydra config, given the config module, name, and overrides.

    :param config_module: Python module path of the hydra configuration package
    :param config_name: Name of the defaults configuration yaml file within `config_module`
    :param overrides: Overrides as kwargs, e.g. env="cartpole", configuration="test"
    :return: Hydra DictConfig instance, assembled according to the given module, name, and overrides.
    """
    with initialize_config_module(config_module):
        cfg = compose(config_name, overrides=overrides)
    return cfg


def init_validation_runner(
    test_levels: list[int], time_limit: int, flat_time_limit: Optional[int] = None
) -> np.ndarray:
    """Instantiates and runs a validation runner with the given time limits and returns the score.
    :param test_levels: the test levels to run the evaluation on.
    :param time_limit: the maximum time limit to be used to limit the evaluation time.
    :param flat_time_limit: the time limit to limit the time taken to run a single flat_step.
    :return: a numpy array with the score for each seed evaluated.
    """
    exp_name = 'multi_train/rollout/heuristic/validation_rollout'
    cfg = read_hydra_config_with_overrides(
        config_module='maze.conf',
        config_name='conf_rollout',
        overrides=[f'+experiment={exp_name}', 'runner.time_limit=1'],
    )
    wrappers = {
        'maze_flatland.wrappers.masking_wrapper.FlatlandMaskingWrapper': {
            'mask_builder': {
                '_target_': 'maze_flatland.env.masking.mask_builder.LogicMaskBuilder',
                'mask_out_dead_ends': 'true',
                'disable_stop_on_switches': 'false',
            },
            'explain_mask': 'false',
        },
        'maze_flatland.wrappers.skipping_wrapper.FlatStepSkippingWrapper': {'do_skipping_in_reset': True},
    }
    runner = ValidationRolloutRunner(
        deterministic=True,
        round_level=1,
        test_levels=test_levels,
        record_event_logs=True,
        time_limit=time_limit,
        flat_step_limit=flat_time_limit,
    )
    runner.run_with(env_config=cfg['env'], wrappers=wrappers, agent=cfg['policy'])
    return runner.stats_df['score'].to_numpy()


def test_validation_rollout_runner_time_limit():
    """Tests the correct throwing of time out error when time exceeds the allowed time limit."""
    scores = init_validation_runner(test_levels=[0, 1], time_limit=1, flat_time_limit=None)
    assert len(scores) < 20


def test_validation_rollout_runner_successful():
    """Tests that the validation rollout runner runs smoothly."""
    scores = init_validation_runner(test_levels=[0], time_limit=3600, flat_time_limit=None)
    assert min(scores) >= 0 and len(scores) == 10


def test_validation_rollout_runner_flat_step_time_limit():
    """Tests that the flat_step time limit is triggered as expected."""
    scores = init_validation_runner(test_levels=[0], time_limit=3600, flat_time_limit=0)
    assert max(scores) == -1 and len(scores) == 10
