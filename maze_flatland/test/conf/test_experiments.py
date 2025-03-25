"""Tests for the hydra config files."""

from __future__ import annotations

import os

import maze_flatland.conf as conf_module
from hydra import compose, initialize_config_module
from maze.core.agent.policy import Policy
from maze.core.utils.config_utils import EnvFactory
from maze.core.utils.factory import Factory
from maze.maze_cli import maze_run
from maze.perception.models.model_composer import BaseModelComposer
from maze.runner import Runner
from maze_flatland.env.maze_env import FlatlandEnvironment
from omegaconf import DictConfig
from torch import nn

module_path_single_train = conf_module.__path__[0] + '/experiment/single_train/'


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


def check_instantiation(cfg: DictConfig) -> None:
    """Check the instantiation of a config."""
    env_factory = EnvFactory(cfg.env, cfg.wrappers if 'wrappers' in cfg else {})
    env = env_factory()
    assert env is not None

    assert isinstance(env, FlatlandEnvironment)

    if 'policy' in cfg:
        Factory(Policy).instantiate(cfg['policy'])

    if 'model' in cfg:
        model_composer = Factory(BaseModelComposer).instantiate(
            cfg.model,
            action_spaces_dict=env.action_spaces_dict,
            observation_spaces_dict=env.observation_spaces_dict,
            agent_counts_dict=env.agent_counts_dict,
        )
        for pp in model_composer.policy.networks.values():
            assert isinstance(pp, nn.Module)

        if model_composer.critic:
            for cc in model_composer.critic.networks.values():
                assert isinstance(cc, nn.Module)

    if 'runner' in cfg:
        Factory(Runner).instantiate(cfg['runner'])


def run_experiment(exp_name: str, config_name: str):
    """Run a simple experiment from the hydra config."""
    cfg = read_hydra_config_with_overrides(
        config_module='maze.conf',
        config_name=config_name,
        overrides=[f'+experiment={exp_name}'],
    )
    maze_run(cfg)


def instantiate_experiment(exp_name: str, config_name: str):
    """test if the experiment can be instantiated"""
    cfg = read_hydra_config_with_overrides(
        config_module='maze.conf',
        config_name=config_name,
        overrides=[f'+experiment={exp_name}'],
    )
    check_instantiation(cfg)


def test_single_agent_rollouts():
    """Search for the greedy rollout, then run and check its configuration."""
    for root, _, files in os.walk(module_path_single_train):
        if 'rollout' in root:
            for file in files:
                if file.endswith('.yaml'):
                    rollout_experiment_config = (
                        os.path.join(root, file).removesuffix('.yaml').removeprefix(conf_module.__path__[0])
                    )
                    rollout_experiment_config = rollout_experiment_config.removeprefix('/experiment/')
                    if 'single_agent_greedy_heuristic' in file:
                        run_experiment(rollout_experiment_config, 'conf_rollout')
                        instantiate_experiment(rollout_experiment_config, 'conf_rollout')


def test_single_agent_training_experiments():
    """Search for experiments file, then check their configurations."""
    for root, _, files in os.walk(module_path_single_train):
        if '/train' in root:
            for file in files:
                if file.endswith('.yaml'):
                    train_experiment_config = (
                        os.path.join(root, file).removesuffix('.yaml').removeprefix(conf_module.__path__[0])
                    )
                    train_experiment_config = train_experiment_config.removeprefix('/experiment/')
                    print(f'[RUNNING] {train_experiment_config}')
                    instantiate_experiment(train_experiment_config, 'conf_train')
