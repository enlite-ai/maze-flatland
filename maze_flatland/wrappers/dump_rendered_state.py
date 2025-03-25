"""Wrapper to dump the rendered state during a rollout."""
from __future__ import annotations

import os.path
import shutil
from typing import Any, Optional, Union

from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.wrappers.wrapper import Wrapper
from maze.utils.bcolors import BColors
from maze_flatland.env.maze_env import FlatlandEnvironment


class DumpRenderedStateWrapper(Wrapper[FlatlandEnvironment]):
    """Dumps step renderings of environments as png files.

    Make sure to activate this only for rollouts and disable it during training (e.g. set export=False).
    Otherwise, it will dump a lot off rollout states to your disk.

    :param env: The environment to wrap.
    :param export: Only if set to True the states are dumped.
    :param out_folder: The target folder used to save the rendered states.
    """

    def __init__(self, env: FlatlandEnvironment, export: bool, out_folder: str = 'rendered_states/'):
        super().__init__(env)
        self._episode_subdir = None
        self._export = export
        self.out_path = out_folder
        self._step_actions = []
        if export:
            BColors.print_colored('Rollout rendering enabled - Evaluation will be slower.', BColors.WARNING)

    @override(BaseEnv)
    def step(self, action: MazeActionType) -> tuple[ObservationType, Any, bool, dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map observation.
        :param action: The action to step the environment with
        :return: A tuple with the observation, the reward and the terminal flat along with an extra info dictionary"""
        do_rendering = not self.is_cloned() and self._export
        if do_rendering:
            # Convert to maze action to get the right action in case hierarchical space is used.
            self._step_actions.append(self.action_conversion.space_to_maze(action, self.get_maze_state()).value)

        if do_rendering and self.actor_id().agent_id == self.n_trains - 1:
            self._render(self._step_actions)
            self._step_actions = []

        observation, reward, done, info = self.env.step(action)

        return observation, reward, done, info

    @override(BaseEnv)
    def reset(self) -> ObservationType:
        """Intercept ``BaseEnv.reset`` and map observation.
        :return: The observation after resetting the environment."""

        # Write out put in reset in order to not store rendering in input dir
        self.out_path = os.path.abspath(self.out_path)

        # reset wrapped env
        observation = self.env.reset()
        do_rendering = not self.is_cloned() and self._export
        if do_rendering:
            # update timestamp of episode
            self._episode_subdir = (
                f'{self.out_path}/rollout_{self.env.context.episode_id}_seed_{self.env.get_current_seed()}'
            )
            if os.path.exists(self._episode_subdir):
                shutil.rmtree(self._episode_subdir)
            os.makedirs(self._episode_subdir)

        return observation

    def _render(self, action: MazeActionType) -> None:
        """Render state to rgb image and append image stack.
        :param action: The action taken in the last step
        """
        self.env.render(action=action, save_in_dir=self._episode_subdir)

    @override(Wrapper)
    def get_observation_and_action_dicts(
        self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType], first_step_in_episode: bool
    ) -> tuple[Optional[dict[Union[int, str], Any]], Optional[dict[Union[int, str], Any]]]:
        raise NotImplementedError

    # pylint: disable=protected-access
    @override(SimulatedEnvMixin)
    def clone_from(self, env: DumpRenderedStateWrapper) -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`.
        :param env: the source environment to be cloned."""
        self.env.clone_from(env)
        self._export = env._export
        self.out_path = env.out_path
        self._episode_subdir = env._episode_subdir
