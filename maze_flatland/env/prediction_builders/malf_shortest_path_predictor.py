"""File holding the predictor for shortest path that considers malfunctioning trains."""
from __future__ import annotations

import math
from typing import Optional

import flatland
import numpy as np
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import TrainState
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.envs.rail_trainrun_data_structures import Waypoint
from maze.core.annotations import override


def _get_agent_position(agent: flatland.envs.rail_env.EnvAgent) -> tuple[int, int]:
    """Accessory method used to retrieve the position of an agent.
    :return: Position of the agent, either accessed or "estimated".
    """
    if agent.state.is_off_map_state():
        pos = agent.initial_position
    elif agent.state.is_on_map_state():
        pos = agent.position
    else:
        assert agent.state == TrainState.DONE, (
            f'agent state is {agent.state}, ' f'but was expected to be {TrainState.DONE}'
        )
        pos = agent.target

    return pos


def _shortest_path_for_agent(
    distance_map,
    agent_position: tuple[int, int],
    agent_direction: int,
    max_depth: int,
    agent_handle: int,
    agent_target: tuple[int, int],
) -> list[Waypoint]:
    """Method used to get the prediction of the shortest path for an agent given
    the starting point, direction and its target.

    :param distance_map: Distance map of the agent based on the current rail.
    :param agent_position: Starting position of the agent.
    :param agent_direction: Direction of the agent.
    :param agent_handle: Handle of the agent.
    :param max_depth: Maximum depth of the path.
    :param agent_target: Target of the agent.

    :return: List of waypoints visited during the path.
    """
    position = agent_position  # Init starting pos
    direction = agent_direction  # Init starting dir
    distance = math.inf
    depth = 0
    shortest_paths = []
    while position != agent_target and depth < max_depth:
        next_actions = get_valid_move_actions_(direction, position, distance_map.rail)
        best_next_action = None
        for next_action in next_actions:
            next_action_distance = distance_map.get()[
                agent_handle, next_action.next_position[0], next_action.next_position[1], next_action.next_direction
            ]
            if next_action_distance < distance:
                best_next_action = next_action
                distance = next_action_distance

        shortest_paths.append(Waypoint(position, direction))
        depth += 1

        # if there is no way to continue, the rail must be disconnected!
        # (or distance map is incorrect)
        if best_next_action is None:
            # Path not connected.
            # If it was connected then we would either reach the target or max_depth.
            shortest_paths = []
            break
        position = best_next_action.next_position
        direction = best_next_action.next_direction
    if depth < max_depth:
        shortest_paths.append(Waypoint(position, direction))

    return shortest_paths


class MalfShortestPathPredictorForRailEnv(PredictionBuilder):
    """
    ShortestPathPredictorForRailEnv object that copes with malfunctioning trains.

    This object returns shortest-path predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment.
    The prediction is based on the action with minimal time required to reach a target destination.

    :param exclude_off_map_trains: Whether to exclude the trains that are currently off map from the prediction.
    :param consider_departure_delay: Whether to take into account the earliest departure time in the estimation.
    """

    def __init__(self, exclude_off_map_trains: bool, consider_departure_delay: bool, max_depth: int = 20):
        assert not (exclude_off_map_trains and consider_departure_delay), (
            'Set exclude_off_map_trains to true to not '
            'estimate the path for waiting trains.\n'
            'Set consider_departure_delay to True if you want'
            'to consider the dealy of these train in the path prediction.'
        )
        super().__init__(max_depth)
        self.exclude_off_map_trains = exclude_off_map_trains
        self.consider_departure_delay = consider_departure_delay
        self.persistent_pred_path = {}

    def _find_idx_pos_of_agent(self, agent_handle: int) -> tuple[int, tuple[int, int], int, list[Waypoint]]:
        """Helper method for get_persistent_shortest_path().
        It iterates through the predicted waypoints until it finds the position matching the current agent pos.
        :return: A triple with the index from where the prediction start,
                 position, direction from the last predicted point,
                 and the piece of prediction that still apply.
        """
        idx_pos = -1
        # retrieve persistent prediction_list from the env.
        prediction_list = self.env.dev_pred_dict.get(agent_handle, [])
        reusable_prediction = []
        # get pos of the agent
        agent_pos = _get_agent_position(self.env.agents[agent_handle])
        # init last_pos and last_direction
        last_pos = agent_pos
        last_direction = self.env.agents[agent_handle].direction
        # get the starting point for the prediction to be computed.
        for idx, pred_pos in enumerate(prediction_list):
            if pred_pos.position == agent_pos:
                idx_pos = idx
                last_element = prediction_list[-1]
                last_pos = last_element.position
                last_direction = last_element.direction
                reusable_prediction = prediction_list[idx_pos:]
                break
        return idx_pos, last_pos, last_direction, reusable_prediction

    def get_persistent_shortest_path(self, agent_handle: int) -> dict[int, Optional[list[Waypoint]]]:
        """Method used to update and store in memory the prediction for the shortest path.
        If an agent has already traveled m cells from the previous prediction,
        the remaining n - m cells are retained, and the path is shifted to the current point.
        Then, the next m cells are newly predicted from the last point and appended to the stored path.

        :param agent_handle: Handle of the agent.
        :return: Dictionary with agent's handle as key and list of waypoints traversed by the agent as value.
        """
        distance_map: DistanceMap = self.env.distance_map
        pred_depth = self.max_depth
        # persistent patg stored in env @ self.env.dev_pred_dict
        idx_pos, agent_pos, agent_direction, reusable_prediction = self._find_idx_pos_of_agent(agent_handle)
        flag_offset_correction = False
        if idx_pos == 0:
            # no update needed. Agent has not moved.
            return {agent_handle: self.env.dev_pred_dict[agent_handle]}
        if idx_pos > 0:
            # 1+ is needed otherwise this would be (idx_pos -1) deep.
            pred_depth = 1 + idx_pos  # update the depth of "search" for future positions
            flag_offset_correction = True  # Trigger flag for offset correction.
        # get the missing piece of prediction needed to have the max_depth prediction.
        continuation_of_prediction = _shortest_path_for_agent(
            distance_map, agent_pos, agent_direction, pred_depth, agent_handle, self.env.agents[agent_handle].target
        )
        if flag_offset_correction:
            # fix the observation considering the offset of 1.
            continuation_of_prediction = continuation_of_prediction[1:]
        shortest_path = reusable_prediction + continuation_of_prediction

        # update the shortest path and store it in the env such as we are covered with cloning.
        self.env.dev_pred_dict[agent_handle] = shortest_path
        return {agent_handle: shortest_path}

    def get(self, handle: int = None) -> dict:
        """
        Called whenever get_many in the observation build is called.
        Requires the environment to be set through set_env().
        Equal to short path predictor defined at flatland.envs.predictions.ShortestPathPredictorForRailEnv
        With the only exception of forcing the agent to be stopped in the current cell when malfunctioning.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
             Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here (not implemented yet)
            The prediction at 0 is the current position, direction etc.
        """
        agents = self.env.agents if handle is None else [self.env.agents[handle]]

        shortest_paths = {}

        for agent in agents:
            # cost linear to number of keys.
            shortest_paths.update(self.get_persistent_shortest_path(agent.handle))

        malfunctions_counter = [agent.malfunction_handler.malfunction_down_counter for agent in agents]
        if self.exclude_off_map_trains and not self.consider_departure_delay:
            prediction_dict = self.get_excluding_off_map_trains(agents, shortest_paths, malfunctions_counter)
        elif self.consider_departure_delay:
            prediction_dict = self.get_with_departing_delay(agents, shortest_paths, malfunctions_counter)
        else:
            prediction_dict = self.get_vanilla(agents, shortest_paths, malfunctions_counter)
        return prediction_dict

    def get_excluding_off_map_trains(
        self, agents: list[any], shortest_paths: dict[int, Optional[list[Waypoint]]], malfunctions_counter: list[int]
    ):
        """Get predictions without predicting positions for off map trains.
        :param agents: List of agents to get the predictions for.
        :param shortest_paths: The shortest paths for each agent.
        :param malfunctions_counter: The time for which each train is malfunctioning.
        :return: Dictionary with the predicted position and direction over time.
        """
        prediction_dict = {}
        for idx, agent in enumerate(agents):
            # if the agent is not in an interesting state do not get the prediction

            if agent.state in [TrainState.WAITING, TrainState.MALFUNCTION_OFF_MAP, TrainState.DONE]:
                prediction = np.zeros(shape=(self.max_depth + 1, 5))
                for i in range(self.max_depth + 1):
                    prediction[i] = [i, None, None, None, None]
                prediction_dict[agent.handle] = prediction
                continue

            agent_virtual_direction = agent.direction
            agent_virtual_position = agent.position
            if agent.state == TrainState.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position

            agent_speed = agent.speed_counter.speed
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))

            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # if there is the shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position
            for index in range(1, self.max_depth + 1):
                if new_position == agent.target and not shortest_path:
                    # if agent is arrived then just remove it from the map.
                    prediction[index] = [index, None, None, None, None]
                    continue
                if not shortest_path:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    continue

                if malfunctions_counter[idx] > 0:
                    malfunctions_counter[idx] -= 1
                    prediction[index] = [index, *new_position, agent.direction, RailEnvActions.STOP_MOVING]
                    continue

                # if fractional speed then stay on the cell.
                if index % times_per_cell == 0:
                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction
                    shortest_path = shortest_path[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]

            prediction_dict[agent.handle] = prediction

        return prediction_dict

    def get_vanilla(
        self, agents: list[any], shortest_paths: dict[int, Optional[list[Waypoint]]], malfunctions_counter: list[int]
    ):
        """Get predictions without predicting positions for off map trains.
        :param agents: List of agents to get the predictions for.
        :param shortest_paths: The shortest paths for each agent.
        :param malfunctions_counter: The time for which each train is malfunctioning.
        :return: Dictionary with the predicted position and direction over time.
        """
        prediction_dict = {}

        for idx, agent in enumerate(agents):
            agent_virtual_position = self._get_agent_virtual_position(agent)
            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_counter.speed
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # if there is the shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position

            for index in range(1, self.max_depth + 1):
                if new_position == agent.target and not shortest_path:
                    # if agent is arrived then just remove it from the map.
                    prediction[index] = [index, None, None, None, None]
                    continue
                if not shortest_path:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    continue

                if malfunctions_counter[idx] > 0:
                    malfunctions_counter[idx] -= 1
                    prediction[index] = [index, *new_position, agent.direction, RailEnvActions.STOP_MOVING]
                    continue

                # if fractional speed then stay on the cell.
                if index % times_per_cell == 0:
                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction
                    shortest_path = shortest_path[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]

            prediction_dict[agent.handle] = prediction
        return prediction_dict

    # pylint: disable=too-many-statements
    def get_with_departing_delay(
        self, agents: list[any], shortest_paths: dict[int, Optional[list[Waypoint]]], malfunctions_counter: list[int]
    ):
        """Get predictions without predicting positions for off map trains.
        :param agents: List of agents to get the predictions for.
        :param shortest_paths: The shortest paths for each agent.
        :param malfunctions_counter: The time for which each train is malfunctioning.
        :return: Dictionary with the predicted position and direction over time.
        """
        prediction_dict = {}

        env_time = self.env._elapsed_steps  # pylint: disable=protected-access
        for idx, agent in enumerate(agents):
            waiting_time = agent.earliest_departure - env_time

            agent_virtual_position = self._get_agent_virtual_position(agent)
            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_counter.speed
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

            # if there is the shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position

            for index in range(1, self.max_depth + 1):
                if new_position == agent.target and not shortest_path:
                    # if agent is arrived then just remove it from the map.
                    prediction[index] = [index, None, None, None, None]
                    continue
                if not shortest_path:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    continue

                if malfunctions_counter[idx] > 0:
                    malfunctions_counter[idx] -= 1
                    prediction[index] = [index, *new_position, agent.direction, RailEnvActions.STOP_MOVING]
                    # if train malfunction outside of map decrease waiting time as well...
                    if waiting_time > 0:
                        waiting_time -= 1
                    continue

                if waiting_time > 0:
                    waiting_time -= 1
                    prediction[index] = [index, *new_position, agent.direction, RailEnvActions.STOP_MOVING]
                    continue

                # if fractional speed then stay on the cell.
                if index % times_per_cell == 0:
                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction
                    shortest_path = shortest_path[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]

            prediction_dict[agent.handle] = prediction
        return prediction_dict

    @classmethod
    def _get_agent_virtual_position(cls, agent) -> tuple[int, int]:
        """Helper method to get the position of an agent based on its state.
        :param agent: The agent to get the position for.
        :return: Tuple with the current position of the agent."""
        if agent.state.is_off_map_state():
            agent_virtual_position = agent.initial_position
        elif agent.state.is_on_map_state():
            agent_virtual_position = agent.position
        else:
            assert agent.state == TrainState.DONE, f'state of agent is not recognised: {agent.state}'
            agent_virtual_position = agent.target
        return agent_virtual_position

    @override(flatland.core.env_prediction_builder.PredictionBuilder)
    def set_env(self, env: flatland.core.env.Environment) -> None:
        """
        Sets environment (also for integrated ShortestPathPredictorForRailEnv).
        """
        super().set_env(env)
