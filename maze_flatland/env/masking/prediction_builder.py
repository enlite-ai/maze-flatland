"""File holding prediction builder for short path computation."""
from __future__ import annotations

import flatland.core.env
import numpy as np
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.observations import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.step_utils.states import TrainState
from maze.core.annotations import override
from maze_flatland.env.maze_action import FlatlandMazeAction


class ShortestPathPredictorBuilder(ObservationBuilder):
    """
    Build observations which indicate the shortest path to the target. Based on
    https://github.com/AIcrowd/flatland-getting-started/blob/master/notebook_2.ipynb.

    Assembles a representation indicating the minimum distance for each of the 3 available directions for each
    agent (Left, Forward, Right).
    """

    def __init__(self):
        super().__init__()
        self._shortest_path_predictor = ShortestPathPredictorForRailEnv(max_depth=1)

    @override(ObservationBuilder)
    def set_env(self, env: flatland.core.env.Environment) -> None:
        """
        Sets environment (also for integrated ShortestPathPredictorForRailEnv).
        """
        super().set_env(env)
        self._shortest_path_predictor.set_env(env)

    @override(flatland.envs.observations.ObservationBuilder)
    def reset(self) -> None:
        """
        Resets shortest path predictor.
        """
        self._shortest_path_predictor.reset()

    def get(self, handle: int = None) -> list[dict[FlatlandMazeAction, float]]:
        agents = self.env.agents if handle is None else [self.env.agents[handle]]
        return [self.get_distances_from_next_position(agent) for agent in agents]

    def get_distances_from_next_position(self, agent: EnvAgent) -> dict[FlatlandMazeAction, int]:
        """Compute the distances from the next possible positions based on the outgoing connections
            of the current cell.

        :param agent: The agent.
        :return: dictionary with actions and the distance to target from the future positions.
        """
        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
            agent_pos = agent.position
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)
            if agent.state == TrainState.WAITING:
                agent_pos = agent.initial_position
            elif agent.state == TrainState.READY_TO_DEPART:
                agent_pos = agent.initial_position
            elif agent.state == TrainState.DONE:
                agent_pos = agent.target
            elif agent.state == TrainState.MALFUNCTION_OFF_MAP:
                agent_pos = agent.initial_position
            else:
                assert False, 'Agent status and position are incompatible for computation of shortest path.'
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = flatland.core.grid.grid4_utils.get_new_position(agent_pos, direction)
                min_distances.append(
                    self.env.distance_map.get()[agent.handle, new_position[0], new_position[1], direction]
                )
            else:
                min_distances.append(np.inf)
        return {
            FlatlandMazeAction.DEVIATE_LEFT: min_distances[0],
            FlatlandMazeAction.GO_FORWARD: min_distances[1],
            FlatlandMazeAction.DEVIATE_RIGHT: min_distances[2],
        }
