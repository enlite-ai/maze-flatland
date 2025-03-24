"""File holding the tests for the renderer functionality."""
from __future__ import annotations

import os
import random

from maze_flatland.env.renderer import FlatlandRendererBase
from maze_flatland.test.env_instantation import create_env_for_testing
from maze_flatland.wrappers.dump_rendered_state import DumpRenderedStateWrapper


def create_wrapped_env(export: bool) -> DumpRenderedStateWrapper:
    """Return a flatland environment wrapped into the DumpRenderedStateWrapper.

    :param export: whether to export the rendered image or not.
    :return: A flatland environment wrapped into the DumpRenderedStateWrapper.
    """
    env = DumpRenderedStateWrapper.wrap(create_env_for_testing(), export=export)
    env.seed(1234)
    _ = env.reset()
    return env


def test_render():
    """Test the render function by checking that after a flat-step exactly one file is created with the
    correct name."""
    random.seed(1234)
    env = create_wrapped_env(True)
    ep_path = os.path.join(env.out_path, os.listdir(env.out_path)[0])
    # step all the agents once
    for _ in range(env.n_trains):
        _ = env.step({'train_move': 0})
    # check that exists the file
    assert os.path.isfile(os.path.join(ep_path, f'render_{0}-{env.n_trains-1}.png')), 'File not found.'
    # check that there are no additional files.
    assert len(os.listdir(ep_path)) == 1


def test_renderer_reset():
    """Test that an initialized renderer is correctly reset when resetting the env."""
    env = create_wrapped_env(True)
    renderer: FlatlandRendererBase = env.get_renderer()
    # check that is not initialized before stepping the env.
    assert not renderer.is_initialized
    for _ in range(env.n_trains):
        _ = env.step({'train_move': 2})
    # check that after stepping, the renderer is correctly init
    assert renderer.is_initialized
    env.reset()
    # check that after reset, the renderer is not initialized.
    assert not renderer.is_initialized
