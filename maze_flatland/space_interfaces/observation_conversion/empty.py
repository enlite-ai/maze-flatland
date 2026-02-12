"""Blank observation space."""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from maze.core.annotations import override
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.observation_conversion.base import BaseObservationConversion


class ObservationConversionBlank(BaseObservationConversion):
    """Blank observation space. Useful for performance reasons when running either a random policy,
    or a policy that acts directly on the maze_state (and can obtain it from the environment via the needs_state
    flag)
    """

    def __init__(self):
        super().__init__(False)

    # pylint: disable=unused-argument
    @override(BaseObservationConversion)
    def maze_to_space(self, maze_state: MazeStateType) -> ObservationType:
        """Return a dummy observation.

        :param maze_state: Reference to the maze_state
        :return: Observation representation of the maze_state
        """
        return {'dummy': np.zeros((0,), dtype=np.float32)}

    @override(BaseObservationConversion)
    def space_to_maze(self, observation: ObservationType) -> FlatlandMazeState:
        """Not implemented."""
        raise NotImplementedError

    def space(self) -> spaces.Dict:
        """Return the dummy observation space.

        :return: Dummy observation space.
        """
        return spaces.Dict({'dummy': spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)})

    def reset(self, core_env: FlatlandCoreEnvironment) -> None:
        """Reset the observation conversion.

        :param core_env: The core environment.
        """
