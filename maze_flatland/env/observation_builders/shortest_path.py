"""
Observation builder computing the shortest path for each individual agent.
"""


from __future__ import annotations

import flatland.core.env
import flatland.core.grid.grid4_utils
import flatland.envs.agent_utils
import flatland.envs.observations
import numpy as np
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from maze.core.annotations import override
from maze_flatland.env.maze_action import FlatlandMazeAction


class ShortestPathObservationBuilder(flatland.envs.observations.ObservationBuilder):
    """
    Build observations which indicate the shortest path to the target. Based on
    https://github.com/AIcrowd/flatland-getting-started/blob/master/notebook_2.ipynb.

    Assembles a representation indicating the minimum distance for each of the 3 available directions for each
    agent (Left, Forward, Right).
    """

    def __init__(self, max_depth: int):
        super().__init__()
        self._shortest_path_predictor = ShortestPathPredictorForRailEnv(max_depth=max_depth)

    @override(flatland.envs.observations.ObservationBuilder)
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

    # pylint: disable=broad-exception-raised
    @override(flatland.envs.observations.ObservationBuilder)
    def get(self, handle: int = 0) -> tuple[dict[FlatlandMazeAction, int], FlatlandMazeAction, np.ndarray]:
        """
        Computes path length for specified agent for each possible direction (see FlatlandMazeAction): leftwards,
        forward, rightwards.
        :param handle: Train ID.
        :return: Tuple of:
            (a) Path distances for left, forward, right in current step. If a direction is not possible, the
                corresponding value will be np.inf.
            (b) Action with shortest entailing path to target.
            (c) Complete shortest path up to specified depth.
        """

        agent: flatland.envs.agent_utils.EnvAgent = self.env.agents[handle]

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = self.env.rail.get_transitions(*agent.initial_position, agent.direction)

        # Get agent's position.
        if self.env.agents[handle].position:
            agent_pos = agent.position
        # If agent hasn't been placed yet: Check for initial/terminal states or raise exception if neither is the case.
        else:
            if agent.state == flatland.envs.agent_utils.TrainState.WAITING:
                agent_pos = agent.initial_position
            elif agent.state == flatland.envs.agent_utils.TrainState.READY_TO_DEPART:
                agent_pos = agent.initial_position
            elif agent.state == flatland.envs.agent_utils.TrainState.DONE:
                agent_pos = agent.target
            elif agent.state == flatland.envs.agent_utils.TrainState.MALFUNCTION_OFF_MAP:
                agent_pos = agent.initial_position
            else:
                raise Exception('Agent status and position are incompatible for computation of shortest path.')

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        min_distances = []
        for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = flatland.core.grid.grid4_utils.get_new_position(agent_pos, direction)
                min_distances.append(self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
            else:
                min_distances.append(np.inf)

        return (
            {
                FlatlandMazeAction.DEVIATE_LEFT: min_distances[0],
                FlatlandMazeAction.GO_FORWARD: min_distances[1],
                FlatlandMazeAction.DEVIATE_RIGHT: min_distances[2],
            },
            # Convert to MazeAction by mapping 0 -> left, 1 -> forward, 2 -> right to 1 -> DEVIATE_LEFT, 2 ->
            # GO_FORWARD, 3 -> DEVIATE_RIGHT.
            FlatlandMazeAction(int(np.argmin(min_distances)) + 1),
            self._shortest_path_predictor.get(handle)[handle],
        )
