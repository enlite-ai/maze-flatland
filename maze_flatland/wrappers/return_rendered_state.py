"""Wrapper to return the rendered state during a rollout."""
from __future__ import annotations

from typing import Any, Optional, Union

import matplotlib.pyplot as plt
from maze.core.annotations import override
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.wrappers.wrapper import Wrapper
from maze_flatland.env.maze_env import FlatlandEnvironment


class InterceptRendererAndReturnWrapper(Wrapper[FlatlandEnvironment]):
    """Returns the renderings of environment to the caller.

    :param env: The environment to wrap.
    """

    def render(self, action: MazeActionType) -> plt.Figure:
        """Render state to rgb image and append image stack.
        :param action: The action taken in the last step
        :return: Plt figure with the rendered state.
        """
        return self.get_renderer()._render(  # pylint: disable=protected-access
            self.get_maze_state(), action, None, rail_env=self.rail_env, close_prior_windows=True
        )

    @override(Wrapper)
    def get_observation_and_action_dicts(
        self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType], first_step_in_episode: bool
    ) -> tuple[Optional[dict[Union[int, str], Any]], Optional[dict[Union[int, str], Any]]]:
        raise NotImplementedError
