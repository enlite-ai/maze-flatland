"""File holding the observation conversion base class."""
from __future__ import annotations

import pickle
from abc import ABC
from typing import Any

import gymnasium as gym
from flatland.core.env_observation_builder import ObservationBuilder
from maze.core.annotations import override
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationConversionInterface, ObservationType
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState


class BaseObservationConversion(ObservationConversionInterface, ABC):
    """Base class for defining the observation conversion in flatland.
    :param serialize_representation: Boolean flag indicating whether
                to serialize the representation built from the builder.
    """

    observation_builders: dict[type[ObservationBuilder], ObservationBuilder] = {}

    _observation_representation: list[dict[type[ObservationBuilder], Any] | None] = []

    def __init__(self, serialize_representation: bool):
        self.observation_builders: dict[type[ObservationBuilder], ObservationBuilder] = {}
        self._spaces = gym.spaces.Dict({})
        self.serialize_representation = serialize_representation

    def reset(self, core_env: FlatlandCoreEnvironment) -> None:
        """Base method to be used to reset the observation builder"""
        for ob in self.observation_builders.values():
            ob.set_env(core_env.rail_env)
            ob.reset()
        self._observation_representation = []

    def serialize_state(self) -> bytes | None:
        """Serialize the state of the current observation conversion."""
        serialised_state = None
        if self.serialize_representation:
            serialised_state = pickle.dumps(self._observation_representation)
        return serialised_state

    def deserialize_state(self, serialised_state: bytes):
        """Restore a serialised state.
        :param serialised_state: Serialised instance of the observation conversion to recover.
        """
        self._observation_representation = []
        if self.serialize_representation:
            self._observation_representation = pickle.loads(serialised_state)

    def _build_observation(self, maze_state: FlatlandMazeState) -> None:
        """Build the observation representation for all the agents.
            Observation is built at the flat_step level for all the agents that still need to take an action.
        :param maze_state: The current maze_state."""
        assert len(self._observation_representation) == 0 or maze_state.current_train_id == 0
        # flat step or need to recompute it
        obs_representation = {
            k: ob.get_many(list(range(maze_state.current_train_id, maze_state.n_trains)))
            for k, ob in self.observation_builders.items()
        }
        self._observation_representation = [
            {k: obs_repr[idx] for k, obs_repr in obs_representation.items()}
            for idx in range(maze_state.current_train_id, maze_state.n_trains)
        ]

    def pop_observation_representation(self):
        """Return the first element of the observation representation."""
        assert len(self._observation_representation) > 0
        return self._observation_representation.pop(0)

    def maze_to_space(self, maze_state: MazeStateType) -> ObservationType:
        """Base Initialization method to create the observation representation.
        :param maze_state: Reference to the maze_state
        :return: Observation representation of the maze_state
        """
        if len(self._observation_representation) == 0 or maze_state.current_train_id == 0:
            self._build_observation(maze_state)
        return {}

    @override(ObservationConversionInterface)
    def space_to_maze(self, observation: ObservationType) -> FlatlandMazeState:
        """
        We do not provide the conversion of space observations to Flatland's global observations as of this time.
        See :py:meth:`~maze.core.env.observation_conversion.ObservationConversionInterface.space_to_maze`.
        """

        raise NotImplementedError
