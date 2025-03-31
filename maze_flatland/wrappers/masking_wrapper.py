"""Holds the wrapper to support masking in flatland."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.utils.factory import Factory
from maze.core.wrappers.wrapper import ObservationWrapper
from maze.utils.bcolors import BColors
from maze_flatland.env.backend_utils import get_transitions_map
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder, TrainLogicMask
from maze_flatland.env.maze_state import MazeTrainState


class FlatlandMaskingWrapper(ObservationWrapper[MazeEnv]):
    """Wrapper class used to create mask based on the current observation.

    :param env: The maze environment to wrap.
    :param mask_builder: Specify the logic to be used to create a mask.
    :param explain_mask: Whether to show the masking motivation.
    """

    def __init__(
        self,
        env: MazeEnv,
        mask_builder: LogicMaskBuilder,
        explain_mask: bool = False,
    ):
        super().__init__(env)

        # Instance to the mask builder
        self.logic_mask_builder = Factory(LogicMaskBuilder).instantiate(mask_builder)
        self.explain_mask = explain_mask
        self._print_color = '\033[95m'  # magenta

        # Update obs space to include mask.
        for k, obs_space in self.observation_spaces_dict.items():
            action_space = self.action_spaces_dict[k].spaces[k]
            assert isinstance(action_space, gym.spaces.Discrete), 'Currently only discrete action spaces are supported.'
            # Add mask to obs space
            self.observation_spaces_dict[k] = gym.spaces.Dict(
                {**obs_space.spaces, f'{k}_mask': gym.spaces.Box(shape=(action_space.n,), low=0, high=1, dtype=bool)}
            )

    def _create_mask(self, train_handle: int | None = None) -> np.ndarray[bool]:
        """Helper method to create the mask for a given, or current, train.

        :param train_handle: Optional train handle to compute the mask for.
                            Default: None. Taken as maze_state.current_train_id.
        :return: The mask as a list of bools.
        """
        if train_handle is None:
            train_handle = self._current_train_id
        # Get train state from maze state
        train_state = self.get_maze_state().trains[train_handle]
        # Extract transition map from the railEnv
        transition_map = get_transitions_map(self._rail_env)
        # Build mask and return
        logic_train_mask = self.logic_mask_builder.create_train_mask(train_state, transition_map)
        boolean_mask = self.action_conversion.to_boolean_mask(logic_train_mask, train_state)
        self.print_mask_explanation(boolean_mask, logic_train_mask, train_state)
        return boolean_mask

    def get_mask_for_flat_step(self) -> list[np.ndarray[bool]]:
        """Computes and returns the masks at the flat step level for all agents.

        :return: A list of masks.
        """
        maze_state = self.get_maze_state()
        # Extract transition map from the railEnv
        transition_map = get_transitions_map(self._rail_env)
        masks = []
        for train_state in maze_state.trains:
            logic_train_mask = self.logic_mask_builder.create_train_mask(train_state, transition_map)
            boolean_mask = self.action_conversion.to_boolean_mask(logic_train_mask, train_state)
            masks.append(boolean_mask)
            self.print_mask_explanation(masks[-1], logic_train_mask, train_state)
        return masks

    def print_mask_explanation(
        self,
        boolean_mask: np.ndarray[bool],
        logic_train_mask: TrainLogicMask,
        train_state: MazeTrainState,
    ) -> None:
        """Method used to print the motivation behind a mask.
        :param boolean_mask: The logic mask related to the action space.
        :param logic_train_mask: The mask holding the explanation.
        :param train_state: The train state to be used for parsing the mask.
        """
        if self.explain_mask:
            # multiply the mask to the action space to get the actions.
            allowed_actions = np.asarray(self.action_conversion.list_actions())[boolean_mask]
            vrb_debug_mask = f'[{train_state.env_time}.{train_state.handle}] Actions allowed: '
            for action in allowed_actions:
                vrb_debug_mask += f'\n\t - {action}'
            vrb_debug_mask += '\n' + logic_train_mask.explain()
            # Should this be dumped to maze_cli.log?
            BColors.print_colored(vrb_debug_mask, self._print_color)

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> Any:
        """Computes the mask and extends the observation with the mask.

        :param observation: The observation to be combined with the mask.
        :return: The extended observation with the mask.
        """
        mask = self._create_mask()
        # Append mask to the current obs.
        observation[self.actor_id().step_key + '_mask'] = mask
        # Append mask to the current maze_state
        self.get_maze_state().action_masks.append(mask)
        return observation

    @override(SimulatedEnvMixin)
    def clone_from(self, env: FlatlandMaskingWrapper) -> None:
        """Overrides the clone_from method.
        :param env: The source env to be cloned."""
        self.env.clone_from(env.env)
        self.explain_mask = env.explain_mask
