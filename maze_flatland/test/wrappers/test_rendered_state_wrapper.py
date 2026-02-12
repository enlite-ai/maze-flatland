"""File holding tests for the InterceptRendererAndReturnWrapper wrapper."""
from __future__ import annotations

from matplotlib import pyplot as plt
from maze_flatland.test.env_instantation import create_env_for_testing
from maze_flatland.wrappers.return_rendered_state import InterceptRendererAndReturnWrapper


def test_wrapper():
    """Runs a rollout and checks the output of the InterceptRendererAndReturnWrapper."""
    env = create_env_for_testing()
    env = InterceptRendererAndReturnWrapper.wrap(env)
    env.seed(1234)
    env.reset()
    render_fig = env.render([])
    assert isinstance(render_fig, plt.Figure)
