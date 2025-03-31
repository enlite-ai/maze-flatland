"""
    Minimal conversion between Flatland's global observations as MazeStates for a single-agent
    scenario
    (see http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/intro_observation_actions.html#tree-observation,
    https://flatland.aicrowd.com/getting-started/env/observations.html) and space actions.
"""

from __future__ import annotations

import gymnasium as gym
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationType
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion

keys_to_ignore = [
    'actions_required',
    'map_size',
    'current_train_id_oh',
    'target_distances',
    'shortest_path_direction_index',
    'actions_required',
    'pos_curr_direction',
    'pos_other_direction',
    'pos_malfunctions',
    'pos_n_departure_ready',
    'pos_speeds',
    'pos_curr_target',
    'pos_other_target',
    'transitions',
]


class MinimalObservationConversion(PositionalObservationConversion):
    """
    ObservationConversion filtering a subset of features from the encoded positional information.
    Based on Flatland's global observations.
    See :py:meth:`positional.ObservationConversion`.
    """

    @override(PositionalObservationConversion)
    def maze_to_space(self, maze_state: FlatlandMazeState) -> ObservationType:
        """
        See :py:meth:`base_observation_conversion.ObservationConversion.maze_to_space` and
        :py:meth: 'PositionalObservationConversion.maze_to_space'.
        Note that Flatland's GlobalObsForRailEnv, whose information is being used here, returns None for a train which
        has reached its goal. This is reflected in the space observation being produced here.
        """

        return {k: v for k, v, in super().maze_to_space(maze_state).items() if k not in keys_to_ignore}

    @override(PositionalObservationConversion)
    def space_to_maze(self, observation: ObservationType) -> FlatlandMazeState:
        """
        We do not provide the conversion of space observations to Flatland's global observations as of this time.
        See :py:meth:`base_observation_conversion.ObservationConversion.space_to_maze`.
        """

        raise NotImplementedError

    @override(PositionalObservationConversion)
    def space(self) -> gym.spaces.Dict:
        """
        Check https://flatland.aicrowd.com/getting-started/env/observations.html for more info.
        See :py:meth:`positional.ObservationConversion.space`.
        """
        full_obs_dict = {**super().space().spaces}
        return gym.spaces.Dict({k: v for k, v in full_obs_dict.items() if k not in keys_to_ignore})
