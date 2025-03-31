"""
Conversion between Flatland's global observations as MazeStates (see
http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/intro_observation_actions.html#tree-observation,
https://flatland.aicrowd.com/getting-started/env/observations.html) and space actions.
"""

from __future__ import annotations

import flatland.core.grid.grid4
import flatland.envs.agent_utils
import flatland.envs.line_generators
import flatland.envs.observations
import flatland.envs.rail_env
import gymnasium as gym
import numpy as np
from flatland.envs.observations import GlobalObsForRailEnv
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationConversionInterface, ObservationType
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.observation_conversion.simple import SimpleObservationConversion


class PositionalObservationConversion(SimpleObservationConversion):
    """
    ObservationConversion encoding positional information, i.e. represents information on a coordinate basis. Based on
    on Flatland's global observations.
    See :py:meth:`base_observation_conversion.ObservationConversion.__init__`.
    """

    def __init__(self, serialize_representation: bool):
        super().__init__(serialize_representation)
        self.observation_builders[GlobalObsForRailEnv] = GlobalObsForRailEnv()
        self._spaces = self._spaces
        assert len(self.observation_builders) == 1

    @override(SimpleObservationConversion)
    def reset(self, core_env: FlatlandCoreEnvironment):
        """Intercepts ~simple.ObservationConversion.reset"""
        super().reset(core_env)
        map_shape = (self._map_height, self._map_width)
        n_directions = len(flatland.core.grid.grid4.Grid4TransitionsEnum)

        self._spaces = gym.spaces.Dict(
            {
                **super().space().spaces,
                # GlobalObsForRailEnv encodes information in a numpy array of shape (map_width, map_height).
                # Initial observations before trains move for the first time are sometimes encoded
                # with an array of -1 each.
                # Once trains have reached their goal, GlobalObsForRailEnv returns None instead of this numpy array.
                # We provide a numpy array of -1 for both cases (except when GlobalObsForRailEnv returns 0 by default,
                # such as for 'pos_n_departure_ready').
                **{
                    # Transitions represent the possible transitions on a map and are encoded each as one of 16 binary
                    # values.
                    # Transition map seems to be identical for each agent, hence sufficient to include it only once.
                    'transitions': gym.spaces.Box(shape=(*map_shape, 16), low=0, high=1, dtype=np.float32),
                    # Active train's direction, position-encoded.
                    'pos_curr_direction': gym.spaces.Box(
                        shape=map_shape, low=-1, high=n_directions - 1, dtype=np.float32
                    ),
                    # Other train' direction, position-encoded.
                    'pos_other_direction': gym.spaces.Box(
                        shape=map_shape, low=-1, high=n_directions - 1, dtype=np.float32
                    ),
                    # Agents' malfunctions, position-encoded.
                    'pos_malfunctions': gym.spaces.Box(
                        shape=map_shape, low=-1, high=self._max_duration, dtype=np.float32
                    ),
                    # Agents' speeds, position-encoded.
                    'pos_speeds': gym.spaces.Box(shape=map_shape, low=-1, high=self._highest_speed, dtype=np.float32),
                    # Number of agents ready to depart from position.
                    'pos_n_departure_ready': gym.spaces.Box(
                        shape=map_shape, low=0, high=self._n_trains, dtype=np.float32
                    ),
                    # Active agent's target.
                    'pos_curr_target': gym.spaces.Box(shape=map_shape, low=0, high=1, dtype=np.float32),
                    # Other agents' target.
                    'pos_other_target': gym.spaces.Box(shape=map_shape, low=0, high=1, dtype=np.float32),
                },
            }
        )

    @override(SimpleObservationConversion)
    def maze_to_space(self, maze_state: FlatlandMazeState) -> ObservationType:
        """
        See :py:meth:`base_observation_conversion.ObservationConversion.maze_to_space`.
        Note that Flatland's GlobalObsForRailEnv, whose information is being used here, returns None for a train which
        has reached its goal. This is reflected in the space observation being produced here.
        """

        base_obs = super().maze_to_space(maze_state)

        # Default array to return when trains haven't moved yet or have reached their target.
        space = self.space()
        defaults = {
            key: np.ones(subspace.shape, dtype=subspace.dtype) * (subspace.low if hasattr(subspace, 'low') else 0)
            for key, subspace in space.spaces.items()
        }

        # Fetch Flatland's global observation (see
        # https://flatland.aicrowd.com/getting-started/env/observations.html#global-observation) and the current RailEnv
        # instance.
        obs = self.pop_observation_representation()[GlobalObsForRailEnv]
        return {
            **base_obs,
            **{
                key: value.astype(np.float32)
                for key, value in {
                    'transitions': obs[0] if obs else defaults['transitions'],
                    'pos_curr_direction': obs[1][:, :, 0] if obs else defaults['pos_curr_direction'],
                    'pos_other_direction': obs[1][:, :, 1] if obs else defaults['pos_other_direction'],
                    'pos_malfunctions': obs[1][:, :, 2] if obs else defaults['pos_malfunctions'],
                    'pos_speeds': obs[1][:, :, 3] if obs else defaults['pos_speeds'],
                    'pos_n_departure_ready': obs[1][:, :, 4] if obs else defaults['pos_n_departure_ready'],
                    'pos_curr_target': obs[2][:, :, 0] if obs else defaults['pos_curr_target'],
                    'pos_other_target': obs[2][:, :, 1] if obs else defaults['pos_other_target'],
                }.items()
            },
        }

    @override(SimpleObservationConversion)
    def space_to_maze(self, observation: ObservationType) -> FlatlandMazeState:
        """
        We do not provide the conversion of space observations to Flatland's global observations as of this time.
        See :py:meth:`base_observation_conversion.ObservationConversion.space_to_maze`.
        """

        raise NotImplementedError

    @override(ObservationConversionInterface)
    def space(self) -> gym.spaces.Dict:
        """
        Check https://flatland.aicrowd.com/getting-started/env/observations.html for more info.
        See :py:meth:`base_observation_conversion.ObservationConversion.space`.
        """

        return self._spaces
