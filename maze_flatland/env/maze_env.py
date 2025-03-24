"""
MazeEnv for FlatlandCoreEnvironment.
"""
from __future__ import annotations

import pickle
from typing import Any, Union

import numpy as np
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface, ActionType
from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.utils.factory import CollectionOfConfigType, Factory
from maze_flatland.env.core_env import FlatlandCoreEnvironment


class FlatlandEnvironment(MazeEnv[FlatlandCoreEnvironment]):
    """
    Environment for Flatland.
    """

    def __init__(
        self,
        core_env: Union[CoreEnv, dict],
        action_conversion: CollectionOfConfigType,
        observation_conversion: CollectionOfConfigType,
    ):
        super().__init__(
            core_env=Factory(FlatlandCoreEnvironment).instantiate(core_env),
            action_conversion_dict=Factory(ActionConversionInterface).instantiate_collection(action_conversion),
            observation_conversion_dict=Factory(ObservationConversionInterface).instantiate_collection(
                observation_conversion
            ),
        )
        # need to call it at init time to generate the obs_space.
        self._reset_obs_conv()
        self._observation_spaces = {k: obs_conv.space() for k, obs_conv in self.observation_conversion_dict.items()}
        self.observation_original = None
        self.initial_env_time = None
        self.last_action = None
        self.last_maze_action = None

    def _reset_obs_conv(self):
        """Helper method to reset the observation spaces."""
        _ = [obs_space.reset(self.core_env) for obs_space in self.observation_conversion_dict.values()]

    @staticmethod
    @override(MazeEnv)
    def get_done_info(done: bool, info: dict[str, str]) -> tuple[bool, bool]:
        """Intercepts and overrides :py:meth:`~maze.core.env.maze_env.MazeEnv.get_done_info`.

        When episode terminates as a consequence of all trains successfully reaching their target,
        the episode is flagged as truncated which stands for episode solved in this context.

        """
        done_terminated, done_truncated = MazeEnv.get_done_info(done, info)
        done_solved = False
        if done and 'Flatland.Done.successful' in info and info['Flatland.Done.successful']:
            done_solved = True
            done_terminated = False
        return done_terminated, done_truncated or done_solved

    @override(MazeEnv)
    def reset(self):
        """Intercepts maze.core.env.maze_env.MazeEnv.reset."""
        self.core_env.context.reset_env_episode()
        maze_state = self.core_env.reset()
        # reset obs_conv
        self._reset_obs_conv()
        self.observation_original = observation = self.observation_conversion.maze_to_space(maze_state)
        self.initial_env_time = self.get_env_time()

        for key, value in observation.items():
            assert not (
                isinstance(value, np.ndarray) and value.dtype == np.float64
            ), f"observation contains numpy arrays with float64, please convert observation '{key}' to float32"

        return observation

    @override(MazeEnv)
    def clone_from(self, env: MazeEnv) -> None:
        """Reset the maze env to the state of the provided env.

        Note, that it also clones the CoreEnv and its member variables including environment context.

        :param env: The environment to clone from.
        """
        self.deserialize_state(env.serialize_state())
        self.core_env.context.clone_from(env.core_env.context)

    @override(MazeEnv)
    def serialize_state(self) -> bytes:
        """Serialize the current env state and return an object that can be used to deserialize the env again."""
        core_env_state = self.core_env.serialize_state()
        maze_env_state = (self.last_action, self.last_maze_action, self.initial_env_time)
        obs_conv_state = {k: obs_conv.serialize_state() for k, obs_conv in self.observation_conversion_dict.items()}
        reward_state = None
        if self.core_env.reward_aggregator is not None:
            reward_state = self.core_env.reward_aggregator.serialize_state()
        return pickle.dumps((core_env_state, maze_env_state, obs_conv_state, reward_state))

    @override(MazeEnv)
    def deserialize_state(self, serialized_state: bytes) -> None:
        """Deserialize the current env from the given env state."""
        (core_env_state, maze_env_state, obs_conv_state, reward_state) = pickle.loads(serialized_state)
        self.core_env.deserialize_state(core_env_state)
        self.last_action, self.last_maze_action, self.initial_env_time = maze_env_state
        if reward_state is not None:
            self.core_env.reward_aggregator.deserialize_state(reward_state)
        for k, oc_state_serialized in obs_conv_state.items():
            self.observation_conversion_dict[k].deserialize_state(oc_state_serialized)

    @override(MazeEnv)
    # pylint: disable=protected-access
    def _step_core_env(self, action: ActionType) -> tuple[float, bool, dict[Any, Any]]:
        """Override the ~.maze_env._step_core_env to track the executed action before conversion.

        :param action: the action the agent wants to take.
        :return: reward, done, info.
        """
        self.core_env.record_action(action=action)
        return super()._step_core_env(action)
