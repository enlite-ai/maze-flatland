"""Test the correct instantiation of the environment config files."""
from __future__ import annotations

import os

import maze_flatland.conf as conf_module
from maze.core.utils.factory import Factory
from maze_flatland.env.maze_env import FlatlandEnvironment
from omegaconf import OmegaConf


def check_env_instantiation(default_config_path: str):
    """Initialise the environment from the experiments config files.
    :param default_config_path: str to the env config to test."""
    # Load the default config.
    # Use OmegaConf (Hydra dependency / predecessor) to support interpolations (when using plain YAML files,
    # config can be loaded through `yaml.safe_load(...)` directly)
    env_config = OmegaConf.load(default_config_path)['env']

    # Instantiate env.
    env = Factory(base_type=FlatlandEnvironment).instantiate(env_config)
    assert isinstance(env, FlatlandEnvironment)


def test_env_instantation():
    """Iterates through the config files and check the correct instantiation of the env."""
    module_path = conf_module.__path__[0] + '/env/'
    for root, _, files in os.walk(module_path):
        for file in files:
            if file.endswith('.yaml'):
                print(f'Testing: {root}/{file}')
                check_env_instantiation(os.path.join(root, file))
