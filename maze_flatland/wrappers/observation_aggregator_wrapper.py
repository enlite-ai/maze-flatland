"""Wrappers that aggregate the key-values within an observation."""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Union

import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import ObservationWrapper


class AggregationMode(Enum):
    """Defines the aggregation mode with the related functions."""

    STACK = np.stack
    CONCATENATE = np.concatenate

    @staticmethod
    def fn_from_string(name: str):
        """Retrieve the function reference from a string"""
        mapping = {
            'stack': AggregationMode.STACK,
            'concatenate': AggregationMode.CONCATENATE,
        }
        assert name in mapping, f'name {name} not in mapping'
        return mapping[name]


class ObservationAggregatorWithPaddingWrapper(ObservationWrapper[MazeEnv]):
    """Wrapper to aggregate multiple key-values into a single one. The resulting space is the aggregation,
        as specified by `aggregation_mode` of the padded obs spaces.

    The aggregation leads to faster computation in the processing with a neural network as the input
    is processed as a whole.

    :param env: The environment to wrap.
    :param exclude_mask_from_aggregation: Bool flag. When set to true, the mask is left as it is.
    :param aggregation_mode: The aggregation mode to use. E.g., np.stack or np.concatenate.
    """

    def __init__(self, env, exclude_mask_from_aggregation: bool, aggregation_mode: Union[Callable, str]):
        self.aggregation_mode = self._init_aggregation_fn(aggregation_mode)
        super().__init__(env)
        self.exclude_mask_from_aggregation = exclude_mask_from_aggregation
        self.obs_max_shape = None
        self._original_observation_space = env.observation_spaces_dict
        self._observation_spaces = self.processed_shape()

    @classmethod
    def _init_aggregation_fn(cls, aggregation_mode: Union[Callable, str]) -> Callable:
        """Helper method to initialize the aggregation function.

        :param aggregation_mode: The aggregation mode to use.
        :return: The initialized aggregation function.
        """
        if isinstance(aggregation_mode, Callable):
            return aggregation_mode
        return AggregationMode.fn_from_string(aggregation_mode)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: ObservationAggregatorWithPaddingWrapper):
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
        self.exclude_mask_from_aggregation = env.exclude_mask_from_aggregation
        self.aggregation_mode = env.aggregation_mode
        self.obs_max_shape = env.obs_max_shape
        self._original_observation_space = env._original_observation_space  # pylint: disable=protected-access
        self._observation_spaces = env.observation_spaces_dict

    def _get_obs_max_shape(self, in_obs: ObservationType) -> tuple[int, ...]:
        """Extract the maximum shape in the observation space.
            if cached, the shape is returned. Otherwise, it is computed, cached and returned.

        :param in_obs: The observation.
        :return: The maximum shape in the observation space.
        """
        if self.obs_max_shape is None:
            obs_arrays = list(in_obs.values())
            if self.exclude_mask_from_aggregation:
                idx_key = list(in_obs.keys()).index(self.actor_id().step_key + '_mask')
                del obs_arrays[idx_key]

            self.obs_max_shape = tuple(
                max(arr.shape[i] if i < arr.ndim else 0 for arr in obs_arrays)
                for i in range(max(arr.ndim for arr in obs_arrays))
            )
        return self.obs_max_shape

    def pad_to_max_shape_and_aggregate(self, original_observation: ObservationType) -> ObservationType:
        """Pad each space to the max shape within the obs space
            and aggregates the multiple key-values into a single one.

        :param original_observation: The (original) observation to pad and aggregate.
        """
        # Get max possible shape.
        n_dims = len(self._get_obs_max_shape(original_observation))

        out_observation = {}  # the observation to return.
        padded_arrays = []  # a list to store padded observation arrays before collapsing into a key-value pair.

        for key, obs in original_observation.items():
            # process mask if needed.
            if self.exclude_mask_from_aggregation and key == self.actor_id().step_key + '_mask':
                out_observation[key] = obs
                continue

            # First: if required, expand the shape
            if obs.ndim < n_dims:
                # expand from -1 to -n.
                obs = np.expand_dims(obs, axis=tuple(range(-1, -(n_dims - obs.ndim) - 1, -1)))

            # Second, get the required padding based on the shape associated with the key.
            pad_width = [
                (0, max_dim - obs.shape[i]) if i < obs.ndim else (0, max_dim)
                for i, max_dim in enumerate(self._get_obs_max_shape(original_observation))
            ]

            # Third, apply padding to uniform the dimensions of the shapes
            padded_arrays.append(np.pad(obs, pad_width, mode='constant'))

        # Last, apply aggregation and clip it to float32.
        out_observation['observation'] = self.aggregation_mode(padded_arrays, axis=0).astype(np.float32)
        return out_observation

    def processed_shape(self) -> dict[str : gym.spaces.Dict]:
        """Helper methods that aggregate the key-values within an observation.

        :return: Gym dictionary with the aggregated observation space.
        """
        aggregated_obs_space = {}
        for step_key, obs_space in self.observation_spaces_dict.items():
            # define mask identifier
            mask_id = step_key + '_mask'
            # init out dictionary for the obs space.
            step_key_aggregated_obs_space = {}
            # create dummy obs to infer the final shape
            processed_dummy_obs = self.pad_to_max_shape_and_aggregate(obs_space.sample())
            # get shapes to extract low and high values
            boundaries = [
                (np.min(sub_obs_space.low), np.max(sub_obs_space.high)) for sub_obs_space in obs_space.spaces.values()
            ]
            if self.exclude_mask_from_aggregation:
                idx_key = list(obs_space.spaces.keys()).index(mask_id)
                del boundaries[idx_key]

            for key, obs_value in processed_dummy_obs.items():
                if self.exclude_mask_from_aggregation and key == mask_id:
                    low = False
                    high = True
                else:
                    low = np.min(boundaries)
                    high = np.max(boundaries)
                step_key_aggregated_obs_space[key] = gym.spaces.Box(
                    shape=obs_value.shape, low=low, high=high, dtype=obs_value.dtype
                )

            # recreate the space for the current step_key
            aggregated_obs_space[step_key] = gym.spaces.Dict(step_key_aggregated_obs_space)
        return aggregated_obs_space

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> dict[Union[int, str], gym.spaces.Space]:
        """Policy observation spaces as dict."""
        return self._observation_spaces

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> Any:
        """Pre-processes observations.

        :param observation: The observation to be pre-processed.
        :return: The pre-processed observation.
        """
        return self.pad_to_max_shape_and_aggregate(observation)

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self):
        """Keep this env compatible with the gym interface by returning the
        observation space of the current policy."""
        policy_id, actor_id = self.core_env.actor_id()
        return self.observation_spaces_dict[policy_id]


class ObservationAggregatorFlattening(ObservationWrapper[MazeEnv]):
    """Wrapper to aggregate multiple key-values into a single one. The resulting space is
        the concatenation of all the observation spaces flattened.

    The aggregation ensures XGBoost can handle any observation by converting each datapoint into a required 1D format.

    :param env: The environment to wrap.
    :param exclude_mask_from_aggregation: Bool flag. When set to true, the mask is left as it is.

    """

    def __init__(self, env, exclude_mask_from_aggregation: bool):
        super().__init__(env)
        self.exclude_mask_from_aggregation = exclude_mask_from_aggregation
        self._original_observation_space = env.observation_spaces_dict
        self._observation_spaces = self.processed_shape()

    @override(SimulatedEnvMixin)
    def clone_from(self, env: ObservationAggregatorFlattening):
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
        self.exclude_mask_from_aggregation = env.exclude_mask_from_aggregation
        self._original_observation_space = env._original_observation_space  # pylint: disable=protected-access
        self._observation_spaces = env.observation_spaces_dict

    def flatten_and_aggregate(self, original_observation: ObservationType) -> ObservationType:
        """Flatten all spaces and aggregates the multiple key-values into a single one.

        :param original_observation: The (original) observation to pad and aggregate.
        """
        # Get max possible shape.
        out_observation = {}  # the observation to return.
        flatten_arrays = []

        for key, obs in original_observation.items():
            # process mask if needed.
            if self.exclude_mask_from_aggregation and key == self.actor_id().step_key + '_mask':
                out_observation[key] = obs
                continue
            # Flatten and append
            flatten_arrays.append(obs.flatten())

        # Last, apply aggregation and clip it to float32.
        out_observation['observation'] = np.concatenate(flatten_arrays).astype(np.float32)
        return out_observation

    def processed_shape(self) -> dict[str : gym.spaces.Dict]:
        """Helper methods that aggregate the key-values within an observation.

        :return: Gym dictionary with the aggregated observation space.
        """
        aggregated_obs_space = {}
        for step_key, obs_space in self.observation_spaces_dict.items():
            # define mask identifier
            mask_id = step_key + '_mask'
            # init out dictionary for the obs space.
            step_key_aggregated_obs_space = {}
            # create dummy obs to infer the final shape
            processed_dummy_obs = self.flatten_and_aggregate(obs_space.sample())
            # get shapes to extract low and high values
            boundaries = [
                (np.min(sub_obs_space.low), np.max(sub_obs_space.high)) for sub_obs_space in obs_space.spaces.values()
            ]
            if self.exclude_mask_from_aggregation:
                idx_key = list(obs_space.spaces.keys()).index(mask_id)
                del boundaries[idx_key]

            for key, obs_value in processed_dummy_obs.items():
                if self.exclude_mask_from_aggregation and key == mask_id:
                    low = False
                    high = True
                else:
                    low = np.min(boundaries)
                    high = np.max(boundaries)
                step_key_aggregated_obs_space[key] = gym.spaces.Box(
                    shape=obs_value.shape, low=low, high=high, dtype=obs_value.dtype
                )

            # recreate the space for the current step_key
            aggregated_obs_space[step_key] = gym.spaces.Dict(step_key_aggregated_obs_space)
        return aggregated_obs_space

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> dict[Union[int, str], gym.spaces.Space]:
        """Policy observation spaces as dict."""
        return self._observation_spaces

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> Any:
        """Pre-processes observations.

        :param observation: The observation to be pre-processed.
        :return: The pre-processed observation.
        """
        return self.flatten_and_aggregate(observation)

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self):
        """Keep this env compatible with the gym interface by returning the
        observation space of the current policy."""
        policy_id, actor_id = self.core_env.actor_id()
        return self.observation_spaces_dict[policy_id]
