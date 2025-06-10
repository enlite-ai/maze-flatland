"""
MazeState for Flatland environment.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import flatland
import flatland.core.grid.grid4_utils
import flatland.envs.agent_utils
import flatland.envs.observations
import flatland.envs.rail_env
import numpy as np
from flatland.core.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnv
from maze_flatland.env.maze_action import FlatlandMazeAction


class _node_visited_status(IntEnum):
    """Helper class to define the status of the possible nodes visited in a graph.

    Possible values are:
    NOT_VISITED: Never encountered this node before;
    VISITING: Node encountered during the current expansion.
        In the chain of calls, one of the parent has visited this node.
    FULLY_VISITED: Node fully expanded, does not have any more children left to evaluate.
    DEAD: Node has been visited and already proved that will never move again.
    SAFE: Node has been visited and already proved that it could, theoretically, move.
    """

    NOT_VISITED = 0
    VISITING = 1
    FULLY_VISITED = 2
    DEAD = 3
    SAFE = 4


def propagate_status(
    train_handle: int, node_visited: list[int], node_relations: dict[int : list[int]], status: _node_visited_status
) -> None:
    """Iterates through the node relations of the current node id and overrides
        the FULLY_VISITED to the given status.

    :param train_handle: the current train handle.
    :param node_visited: Status of the nodes that have been visited.
    :param node_relations: Relations for blocked trains.
    :param status: Status to be overridden.
    """
    assert status in (_node_visited_status.DEAD, _node_visited_status.SAFE)
    recursive_nodes_to_check = node_relations[train_handle]
    for tid in recursive_nodes_to_check:
        if node_visited[tid] != _node_visited_status.FULLY_VISITED:
            if status == _node_visited_status.DEAD:
                assert node_visited[tid] in (_node_visited_status.DEAD, _node_visited_status.VISITING)
            else:
                assert node_visited[tid] in (_node_visited_status.SAFE, _node_visited_status.VISITING)
            continue
        # override the status
        node_visited[tid] = status
        propagate_status(tid, node_visited, node_relations, status)


def cycling_block_identifier(train_id: int, node_visited: list[int], node_relations: dict[int : list[int]]) -> int:
    """Identifies cycling blocks aka deadlocks through recursive calls.

    It recursively explores the obstructing node_relations between trains to identify if
    a cycle exists. It updates the node_visited list to track the state of each train as:
        - safe: If, at a certain point, it will be able to move;
        - visited: If a train is part of the current chain of control;
        - deadlocked: If the train has already been tested and flagged as "dead".

    :param train_id:  The id of the train for which deadlock detection is performed.
    :param node_relations: Dictionary holding the mapping of blocked trains to blocking trains.
    :param node_visited: List of nodes, trains ids in this case, already visited/checked.

    :return: Status of a train.
    """
    if train_id not in node_relations:
        # if train not blocked, then it is not a deadlock
        node_visited[train_id] = _node_visited_status.SAFE  # Flag as visited and safe
    if node_visited[train_id] != _node_visited_status.NOT_VISITED:
        return node_visited[train_id]

    # Flag as visited (currently exploring this node)
    node_visited[train_id] = _node_visited_status.VISITING
    # check nodes linked recursively
    recursive_nodes_to_check = node_relations[train_id]
    # init flag
    trains_status = [
        cycling_block_identifier(tid_blocking_train, node_visited, node_relations)
        for tid_blocking_train in recursive_nodes_to_check
    ]
    # if not dead stop computation
    tmp_train_is_safe = _node_visited_status.SAFE in trains_status
    tmp_train_is_dead = _node_visited_status.DEAD in trains_status
    train_related_status = np.asarray(node_visited)[recursive_nodes_to_check]
    if (
        np.all(train_related_status == _node_visited_status.FULLY_VISITED)
        or tmp_train_is_dead
        and _node_visited_status.FULLY_VISITED in train_related_status
    ):
        # if all visiting then all have circular dependency, hence dead.
        tmp_train_is_dead = True
        propagate_status(train_id, node_visited, node_relations, _node_visited_status.DEAD)
    elif tmp_train_is_safe and _node_visited_status.FULLY_VISITED in train_related_status:
        # the circular dependency can be broken by at least a train, hence all safe.
        propagate_status(train_id, node_visited, node_relations, _node_visited_status.SAFE)

    if tmp_train_is_safe:
        node_visited[train_id] = _node_visited_status.SAFE
    elif tmp_train_is_dead:
        node_visited[train_id] = _node_visited_status.DEAD
    else:
        node_visited[train_id] = _node_visited_status.FULLY_VISITED
    return node_visited[train_id]


def detect_deadlocks(train_state: list[MazeTrainState]):
    """Detect deadlocks and set the flags.
    :param train_state: The list of maze train states.
    """
    node_relations = detect_blocks_and_obstructions(train_state)
    node_visited = [_node_visited_status.NOT_VISITED for _ in train_state]
    for train in train_state:
        node_status = cycling_block_identifier(train.handle, node_visited, node_relations)
        assert node_status != _node_visited_status.FULLY_VISITED, (
            f'FULLY_VISITED for train {train.handle} ' f'should be resolved through propagation.'
        )
        train.deadlock = _node_visited_status.DEAD == node_status


def detect_blocks_and_obstructions(train_states: list[MazeTrainState]) -> dict[int, list[int]]:
    """Detects (and flag is found) blocks and obstructions for trains.
    :param train_states: List of states of the trains.
    :return: A dictionary mapping blocked trains to those that are obstructing its directions.

    Note: This method is used to identify trains that are obstructed.
        If a train has all its possible directions obstructed then this is blocked.
        If exist a cyclic relation between N trains, then these are in a deadlock
        let's say |b| to be: blocked by where t1 |b| (t2,t3) means t1 is obstructed by t2 and t3.
        Then, we can say that if: t0 |b| (ti) |b| ... |b| t0 -> deadlock.
    """
    # Init variable for the obstructions
    blocked_train_ids = set(range(len(train_states)))
    obstructed_by = {tid: [] for tid in blocked_train_ids}
    # Build a map of agents in each position.
    train_locations = {}
    # First - get location of trains on map
    for t_idx, train in enumerate(train_states):
        if train.is_on_map():
            # update train location.
            train_locations[tuple(train.position)] = t_idx

    # Second - once all locations are updated, then look for obstructions:
    for t_idx, train in enumerate(train_states):
        # skip trains done or that are waiting.
        if train.status in [TrainState.WAITING, TrainState.MALFUNCTION_OFF_MAP, TrainState.DONE]:
            blocked_train_ids.remove(t_idx)
            continue

        # Search any train that can still move.
        if train.status is TrainState.READY_TO_DEPART:
            # A train that is departing want to be placed on map on the exact same cell.
            desired_cell = train.position
            if desired_cell in train_locations:
                idx_blocking_train = train_locations[desired_cell]
                # Deviate Left and Right will never be safe if train is READY_TO_DEPART.
                train.actions_state[FlatlandMazeAction.GO_FORWARD].update_block(idx_blocking_train)
                obstructed_by[train.handle].append(idx_blocking_train)
        else:
            # Otherwise identify obstructions by iterating over the actions.
            for action_state in train.actions_state.values():
                desired_cell = action_state.target_cell
                if desired_cell in train_locations:
                    idx_blocking_train = train_locations[desired_cell]
                    # Set the action as obstructed and the id of the train.
                    action_state.update_block(idx_blocking_train)
                    # update local dict to search for deadlocks.
                    obstructed_by[t_idx].append(idx_blocking_train)

        # Update blocks
        if not train.is_block:
            blocked_train_ids.remove(t_idx)

    # Third and last, create the relations for the blocked trains.
    node_relations = {}
    while len(blocked_train_ids) > 0:
        blocked_train = blocked_train_ids.pop()
        current_train_blocked_by = obstructed_by[blocked_train]
        node_relations[blocked_train] = current_train_blocked_by
    return node_relations


def _fetch_geodesic_distance(
    handle: int,
    position: tuple[int, int],
    direction: int,
    distance_map: flatland.envs.rail_env.DistanceMap,
) -> np.float:
    """
    Fetch geodesic distances from a cell and a direction.
    :param handle: The handle of the id in the rail_env.
    :param position: The position of an agent.
    :param direction: The direction of an agent.
    :param distance_map: The distance map of the rail env.
    """
    distance = distance_map.get()[handle, position[0], position[1], direction]
    return distance


def get_future_direction(action: FlatlandMazeAction, direction: int) -> int:
    """Returns the future direction of the train based on the action given.
    :param action: The action that will be taken by the agent.
    :param direction: The direction of the train.
    """
    assert action in [
        FlatlandMazeAction.GO_FORWARD,
        FlatlandMazeAction.DEVIATE_RIGHT,
        FlatlandMazeAction.DEVIATE_LEFT,
    ]
    if action == FlatlandMazeAction.DEVIATE_LEFT:
        return (direction - 1) % 4
    if action == FlatlandMazeAction.DEVIATE_RIGHT:
        return (direction + 1) % 4
    return direction


def get_next_cell(
    action: FlatlandMazeAction, transitions: tuple[int, int, int, int], position: tuple[int, int], direction: int
) -> tuple[int, int] | None:
    """From the transition of the current cell and an action, it estimates the reachable cell.
    :param action: The action that the agent could take.
    :param transitions: The transition of the current cell.
    :param position: The position of the current cell.
    :param direction: The direction of the train within the cell.
    :return: Next position based on the action.
    """
    action_to_transition_mapping = {
        FlatlandMazeAction.DEVIATE_LEFT: 0,
        FlatlandMazeAction.GO_FORWARD: 1,
        FlatlandMazeAction.DEVIATE_RIGHT: 2,
    }

    possible_transition = transitions[action_to_transition_mapping[action]]

    if possible_transition:
        new_pos = flatland.core.grid.grid4_utils.get_new_position(position, direction)
    else:
        new_pos = None
    return new_pos


class ActionState:
    """Class used to store the action parameters.
    :param handle: Handle of the train.
    :param position: Pos of the train.
    :param direction: Direction of the train.
    :param transitions: Transitions of the current cell occupied by the train.
    :param action: Possible action to be considered.
    :param rail_env: RailEnvironment reference.
    """

    def __init__(
        self,
        handle: int,
        position: tuple[int, int],
        direction: int,
        transitions: tuple[int, int, int, int],
        action: FlatlandMazeAction,
        rail_env: RailEnv,
    ):
        distance_map = rail_env.distance_map
        self.obstructed_by = None
        self.direction = get_future_direction(action, direction)
        self.target_cell = get_next_cell(action, transitions, position, self.direction)
        self.goal_distance = np.inf
        if self.target_cell is not None:
            try:
                distance = _fetch_geodesic_distance(handle, self.target_cell, self.direction, distance_map)
            except IndexError:
                # catch case target on border.
                distance = np.inf
            self.goal_distance = distance

    @property
    def dead_end(self):
        """Returns true if the path is not connected to the target."""
        return np.isinf(self.goal_distance)

    @property
    def obstructed(self):
        """Return true if the agent is obstructed. by another agent."""
        return self.obstructed_by is not None

    def update_block(self, train_idx: int) -> None:
        """Method used to update the block fields.
        :param train_idx: Index of the agent that obstructs the current train.
        """
        assert not self.obstructed
        self.obstructed_by = train_idx

    def is_safe(self):
        """Return whether the action is safe or not.
        An action is safe iff exist a path connecting the target cell to the destination cell.
        """
        return not self.dead_end


class MazeTrainState:
    """Class used to store the internal status of an agent.
    :param handle: Idx of the agent.
    :param old_action: Previous action taken by the agent.
    :param rail_env: Rail backend environment.
    """

    def __init__(self, handle: int, old_action: FlatlandMazeAction, rail_env: flatland.envs.rail_env) -> None:
        train = rail_env.agents[handle]
        self.handle = train.handle
        self.env_time = rail_env._elapsed_steps
        self.max_episode_steps = rail_env._max_episode_steps
        self.status = train.state  # old trains_status
        self.current_speed = train.speed_counter.speed
        self.speed = train.speed_counter.max_speed  # old trains_speeds
        self.target = train.target  # old trains_targets
        self.initial_position = train.initial_position  # old trains_initial_positions
        self.direction = train.direction  # old trains_direction
        self.position = train.position if self.is_on_map() else self.infer_position()
        self.cell_int_id = rail_env.rail.grid[self.position[0], self.position[1]]  # int id of the current cell.
        self.earliest_departure = train.earliest_departure  # old earliest_departure
        self.latest_arrival = train.latest_arrival  # old latest_arrival
        self.malfunction_time_left = train.malfunction_handler.malfunction_down_counter  # old trains_malfunctions
        self.entering_cell = train.speed_counter.is_cell_entry
        self.arrival_time = train.arrival_time
        self.last_action = old_action  # old last_modifying_actions_per_train
        self.deadlock = False
        self.can_stop = False

        # tuple of 4 elements: (left, fw, right, reverse direction).
        self.possible_transition = get_switches_agent_oriented(
            self.direction, rail_env.rail.get_transitions(self.position[0], self.position[1], self.direction)
        )

        self.actions_state = {
            a: ActionState(self.handle, self.position, self.direction, self.possible_transition, a, rail_env)
            for a in (FlatlandMazeAction.DEVIATE_LEFT, FlatlandMazeAction.GO_FORWARD, FlatlandMazeAction.DEVIATE_RIGHT)
        }
        # if a train does not have a solution from the origin.
        self.unsolvable = np.isinf(self.target_distance) and self.has_not_yet_departed()

    @property
    def target_distance(self) -> int:
        """Returns the current distance to the target as number of cells to cross to reach the destination.
        :return: Distance to the target as number of cells to cross to reach the destination.
        """
        if self.is_done():
            return 0
        return min(action_state.goal_distance for action_state in self.actions_state.values()) + 1

    @property
    def out_of_time(self) -> bool:
        """Returns true if the train cannot reach the destination in time.
        :return: True if the train cannot reach the destination in time.
        """
        if self.is_done():
            return False
        time_left = self.max_episode_steps - self.env_time
        return self.best_travel_time_to_target > time_left

    @property
    def best_travel_time_to_target(self) -> int:
        """Estimates the optimal travel time to reach the target."""
        return self.target_distance / self.speed

    @property
    def is_block(self) -> bool:
        """Check if a train is blocked or not.
            A train is blocked if the connected cells are taken by another train.

        :return: True if is blocked, false otherwise
        """
        action_obstructed_flag = [
            action_state.obstructed
            for action_state in self.actions_state.values()
            if action_state.target_cell is not None
        ]
        on_map_or_ready = self.is_on_map() or self.status == TrainState.READY_TO_DEPART
        return on_map_or_ready and all(action_obstructed_flag) and len(action_obstructed_flag) > 0

    def get_action_for_direction(self, global_direction: int) -> FlatlandMazeAction:
        """Returns the action needed to follow a certain global direction.
        :return: FlatlandMazeAction object to transition to the new global direction.
        """
        assert 0 <= global_direction < 4
        if self.direction == global_direction:
            return FlatlandMazeAction.GO_FORWARD
        if global_direction == (1 + self.direction) % 4:
            return FlatlandMazeAction.DEVIATE_RIGHT
        assert global_direction == (self.direction - 1) % 4
        return FlatlandMazeAction.DEVIATE_LEFT

    @property
    def in_transition(self) -> bool:
        """Returns true if the agent is "secretly" transitioning within a cell."""
        fractional_speed = self.speed < 1
        stop_action_given = self.last_action == FlatlandMazeAction.STOP_MOVING
        return not self.entering_cell and fractional_speed and not stop_action_given and not self.deadlock

    @property
    def time_left_to_scheduled_arrival(self) -> int:
        """Method used to calculate the time left until the agent is expected to arrive.
        If negative value then the train is already late.

        :return: 0 if the agent is arrived already otherwise it returns the time left for it.
        """
        if self.is_done():
            return 0
        return self.latest_arrival - self.env_time

    @property
    def arrival_delay(self) -> int:
        """Returns the delay at arrival if a train has already arrived, 0 otherwise."""
        if not self.is_done():
            return 0
        return max(0, self.arrival_time - self.latest_arrival)

    def infer_position(self) -> tuple[int, int]:
        """Given that agent is not on track, it infers the current position based on the state."""
        assert not self.is_on_map()
        if self.is_done():
            return self.target
        assert self.has_not_yet_departed()
        return self.initial_position

    def is_on_map(self) -> bool:
        """Returns true if the train is on the rail.
        :return: True if the train is on the rail, False otherwise."""
        return self.status in (TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION)

    def is_done(self) -> bool:
        """Returns true if the train is already arrived.
        :return: True if the train is already arrived, False otherwise.
        """
        return self.status == TrainState.DONE

    def has_not_yet_departed(self) -> bool:
        """Returns true if the train has not departed yet.
        :return: True if the train has not departed, False otherwise.
        """
        return self.status in [TrainState.READY_TO_DEPART, TrainState.WAITING, TrainState.MALFUNCTION_OFF_MAP]

    @property
    def dead_ends(self) -> list[FlatlandMazeAction]:
        """Return a list of actions leading to a dead end."""
        return [action for action, state in self.actions_state.items() if state.dead_end]

    def should_reverse_dir(self) -> bool:
        """Return whether the train is standing on a reversible dead-end and therefore should revers its direction.

        :return: True if the train is standing on a reversible dead-end, False otherwise.
        """
        return RailEnvTransitionsEnum.is_deadend(self.cell_int_id)


def get_switches_agent_oriented(agent_dir: int, global_switches: tuple[bool]) -> tuple[Any, ...]:
    """Converts the switch of a cell from a global perspective to an agent pov.
    :param agent_dir: Direction of the agent.
    :param global_switches: Possible switches from the global perspective (0 -> up, 1 -> right, 2 -> down...)
    :return: Switches of the current cell considering the agent orientation.
    """
    return global_switches[agent_dir - 1 :] + global_switches[: agent_dir - 1]


class FlatlandMazeState:
    """
    MazeState for Flatland environment.

    Observation builders are Flatland's approach to make the complexity of the environment more manageable by adding a
    layer of abstraction on top that extracts only relevant information as base for a space observation. Within Maze
    this is not necessary, but can be helpful. When deciding whether to build an ObservationBuilder (OB) or to implement
    the extraction of information directly in FlatlandMazeState, consider that OBs can be helpful under one of the
    following conditions:
        - The extraction code is long and complex. OBs provide modularity and can make the code more readable.
        - There are several OBs available, some of them directly via the Flatland API. They can be used in a
          plug-and-play manner.
        - The information extraction code should not be run in all environment configurations, e.g. because it is
          costly. This can be reimplemented in FlatlandMazeState as well, but Maze' configuration abilities already
          offer a way to achieve this.

    :param current_train_id: ID of currently active train.
    :param last_modifying_actions_per_train: Last modifying actions per train (i.e. without DO_NOTHING).
    :param rail_env: Flatland's RailEnv.
    """

    def __init__(
        self,
        current_train_id: int,
        last_modifying_actions_per_train: dict[int, FlatlandMazeAction],
        rail_env: flatland.envs.rail_env.RailEnv,
    ):
        self.trains = [
            MazeTrainState(idx, last_action, rail_env) for idx, last_action in last_modifying_actions_per_train.items()
        ]
        # Filled afterward if needed.
        self.action_masks = []

        self.current_train_id = current_train_id
        # update the blocks and deadlocks fields.

        detect_deadlocks(self.trains)
        self.terminate_episode = False
        self.map_size = (rail_env.height, rail_env.width)

    @property
    def env_time(self):
        """Returns the current time in steps.
        :return: Current time in steps.
        """
        return self.trains[0].env_time

    @property
    def max_episode_steps(self):
        """Returns the maximum episode step.
        :return: The maximum episode step.
        """
        return self.trains[0].max_episode_steps

    @property
    def n_trains(self):
        """Returns the number of trains in the environment.
        :return: Number of trains in the environment.
        """
        return len(self.trains)

    def all_possible_trains_arrived(self) -> bool:
        """Return true if in the current state all trains that can arrive did arrive at their destination."""
        done_trains = np.asarray([train.status == TrainState.DONE for train in self.trains])
        trains_unsolvable = np.asarray([train.unsolvable for train in self.trains])
        if all(np.logical_or(done_trains, trains_unsolvable)):
            return True
        return False

    def current_train(self) -> MazeTrainState:
        """Get the current train state.

        :return: The current maze train state.
        """
        return self.trains[self.current_train_id]

    def current_mask(self) -> np.ndarray:
        """Get the maks for the current state.

        :return: The current mask.
        """
        return self.action_masks[self.current_train_id]
