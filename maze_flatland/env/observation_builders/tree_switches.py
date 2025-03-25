"""
    Enhanced tree observation for flatland from flatland.envs.observations.TreeObsForRailEnv.

    It extends the base observation by collapsing the tree into a graph where the vertices are the switch-cells.
    Furthermore, it provides Idx of deadlock trains encountered as well as
    considering alternative routes for conflicting agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import flatland
import numpy as np
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position, distance_on_rail, position_to_coordinate
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero, fast_position_equal
from flatland.envs.step_utils.states import TrainState
from flatland.utils.ordered_set import OrderedSet
from maze_flatland.env.fast_methods import fast_delete, fast_where
from maze_flatland.env.maze_state import _fetch_geodesic_distance
from maze_flatland.env.prediction_builders.malf_shortest_path_predictor import _get_agent_position

# pylint: disable=too-many-lines


class CellType(IntEnum):
    """
    Possible types of last visited cell w.r.t. the visiting agent.
    """

    UNDEFINED = 0  # type of cell not yet defined.
    SWITCH = 1  # cells holds a switch.
    DEAD_END = 2  # cells has no exit point
    TERMINAL = 3  # cell is already visited or something is wrong, i.e., derail.
    TARGET = 4  # cell holds a target.


@dataclass
class Vertex:
    """
    Vertex class holding features for the vertex.

    A vertex is an abstraction of a switch on the network.
    :param depth: depth of the graph from root.
    :param node: Tree-node of the network related to the vertex.
    :param edges: Unordered list of vertices that can be reached with a distance of 1 in the graph.
    """

    depth: int
    node: Node
    edges: list[Vertex]


@dataclass
class Node:
    """
    Class for storing the node features. Extends the base Node defined at flatland.envs.observations
    Each node information is composed of 14 features.

    :param dist_own_target_encountered: if own target lies on the explored branch the current distance
                                        from the original agent in number of cells is stored.
    :param dist_other_target_encountered: if another agents target is detected the distance in number of cells
                                        from the agents current location is stored
    :param dist_other_agent_encountered: if another agent has been detected, store the distance in number of cells
                                        from current agent position
    :param dist_potential_conflict: Potential conflict consts of two agents traversing the same cell simultaneously.
                                    If that is forecasted, then we store the distance in number of cells from
                                    current agent position.
                                    Otherwise, np.inf = No other agent reserve the same cell at similar time
    :param idx_conflicting_agent: Idx of the agent creating the potential conflict.
    :param cell_of_conflict: Tuple holding the coordinate of the expected conflict location.
    :param dist_other_agent_to_conflict: Distance of the conflicting agent to the cell of conflict.
                                         When not possible to compute it, it is approximated via Manhattan distance.
    :param clashing_agent_has_alternative: Indicates whether the conflicting agent has an alternative.
                                            If no other possibilities then -> Certain conflict(?).
    :param ca_expected_delay_for_alternative_path: If the clashing agent has an alternative, then
                                                    it stores the optimistic delay following the best alternative.
    :param conflict_on_junction_with_multiple_path: Indicates whether the conflict will take place on a switch and
                                                    the selected train has a free path to follow.
                                                    Hence, not going to be blocked nor dead.
    :param expected_delay_for_2nd_best_path: Indicates the delay for resolving the conflict on the switch by detouring.
    :param dist_unusable_switch: if a not not-facing switch (for agent) is detected we store the distance.
    :param unusable_switch_usable_for_ca: Indicates whether the not-facing switch can be used by the conflicting agent.
    :param unusable_switch_delay_for_ca: Indicates the expected delay for the conflicting agent to use
                                        the not-facing switch.
    :param dist_to_next_branch: the distance in number of cells to the next branching  (current node)
    :param dist_min_to_target: minimum distance from node to the agent's target given the direction of the agent
                                if this path is chosen
    :param num_agents_same_direction: number of agents with same direction
    :param num_agents_opposite_direction: number of agents coming from other directions than myself
                                          (so potential conflict)
    :param num_agents_malfunctioning: number of time steps the observed agent remains blocked
    :param speed_min_fractional: slowest observed speed of an agent in same direction. (1 if no agent is observed)
    :param num_agents_ready_to_depart: number of agents ready to depart but not yet active on the node.
    :param idx_deadlock_trains:  idx of the closest encountered train in a deadlock.
    :param children: children node that can be reached from the parent.

    Missing/padding nodes are filled in with -inf (truncated).
    Missing values in present node are filled in with +inf (truncated).

    In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
    In case the target node is reached, the values are [0, 0, 0, 0, 0].
    """

    dist_own_target_encountered: int
    dist_other_target_encountered: int
    dist_other_agent_encountered: int
    dist_potential_conflict: int
    idx_conflicting_agent: int
    cell_of_conflict: tuple[int, int]
    dist_other_agent_to_conflict: int
    clashing_agent_has_alternative: None | bool
    ca_expected_delay_for_alternative_path: int
    conflict_on_junction_with_multiple_path: None | bool
    expected_delay_for_2nd_best_path: int
    dist_unusable_switch: int
    unusable_switch_usable_for_ca: int
    unusable_switch_delay_for_ca: int
    dist_to_next_branch: int
    dist_min_to_target: int
    num_agents_same_direction: int
    num_agents_opposite_direction: int
    num_agents_malfunctioning: int
    speed_min_fractional: float
    num_agents_ready_to_depart: int
    idx_deadlock_trains: list[int]
    children: dict[str, Node]


class TreeSwitchObsRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.
    For details about the features in the tree observation see the get() function.

    :param graph_depth: the depth of the graph obs.
    :param predictor: the predictor used to get the shortest path.
    """

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, graph_depth: int, predictor: PredictionBuilder = None):
        super().__init__()

        self.max_depth = graph_depth
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.location_has_target = None
        self.deadlock_flags = None
        self._flag_init_deadlock = False
        self._nodes_expanding: list[tuple[Vertex, str]] = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = None
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}
        self.location_has_agent_deadlocked = {}

    def reset(self):
        """Reset the observation builder."""
        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}
        self._flag_init_deadlock = False
        self._nodes_expanding: list[tuple[Vertex, str]] = []

    def set_deadlock_trains(self, deadlock_flags: list[bool]) -> None:
        """Initialise the deadlock trains based on the given idxs.

        :param deadlock_flags: Boolean vector for the deadlocked trains.
        """
        assert not self._flag_init_deadlock, 'idx_deadlock_trains already filled.'
        self.deadlock_flags = deadlock_flags
        self._flag_init_deadlock = True

    # pylint: disable=protected-access
    def get_many(self, handles: Optional[list[int]] = None) -> dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.

        :param handles: Optional list of trains id to compute the observation for.
                        If none, then all trains are considered.
        :return: A dict holding the observation for the handles specified by `handles`.
        """
        assert self._flag_init_deadlock, 'Need to set the deadlock trains in advance'
        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        if self.env.agents[a].state == TrainState.DONE:
                            pos_list.append((np.nan, np.nan))
                            dir_list.append(np.nan)
                        else:
                            pos_list.append(self.predictions[a][t][1:3])
                            dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Look-up tables
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}
        self.location_has_agent_deadlocked = {}

        # Update local lookup table for all agents' positions
        # record agents at different positions.
        for train_idx, _agent in enumerate(self.env.agents):
            agent_is_off_map = _agent.state.is_off_map_state()
            # Skip if off map or done. If done then _agent.position = None
            if not agent_is_off_map and _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_counter.speed
                if self.deadlock_flags[train_idx]:
                    self.location_has_agent_deadlocked[tuple(_agent.position)] = train_idx
                self.location_has_agent_malfunction[
                    tuple(_agent.position)
                ] = _agent.malfunction_handler.malfunction_down_counter

            if agent_is_off_map and _agent.initial_position:
                self.location_has_agent_ready_to_depart.setdefault(tuple(_agent.initial_position), 0)
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] += 1

        observations = super().get_many(handles)
        # reset the deadlock status for future calls.
        self._flag_init_deadlock = False
        self.deadlock_flags = []
        return observations

    def get(self, handle: int = 0) -> tuple[Node, Vertex] | None:
        """
        Computes the current observation for agent `handle` in env

        :param handle: idx of the agent to compute observations for.
        :return: tuple[Node, Vertex], node is the base tree observation, Vertex is the graph observation.
        """
        assert handle < len(self.env.agents), f'ERROR: obs _get - handle {handle}, len(agents): {len(self.env.agents)}'
        agent = self.env.agents[handle]
        agent_virtual_position = _get_agent_position(agent)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        # was referring to TreeObsForRailEnv.Node
        root_node_observation = Node(
            dist_own_target_encountered=-1,
            dist_other_target_encountered=-1,
            dist_other_agent_encountered=-1,
            dist_potential_conflict=-1,
            idx_conflicting_agent=-1,
            cell_of_conflict=(-1, -1),
            dist_other_agent_to_conflict=-1,
            clashing_agent_has_alternative=None,
            ca_expected_delay_for_alternative_path=-1,
            conflict_on_junction_with_multiple_path=None,
            expected_delay_for_2nd_best_path=-1,
            dist_unusable_switch=-1,
            unusable_switch_usable_for_ca=-1,
            unusable_switch_delay_for_ca=-1,
            dist_to_next_branch=-1,
            dist_min_to_target=distance_map[(handle, *agent_virtual_position, agent.direction)],
            num_agents_same_direction=0,
            num_agents_opposite_direction=0,
            num_agents_malfunctioning=agent.malfunction_handler.malfunction_down_counter,
            speed_min_fractional=agent.speed_counter.speed,
            num_agents_ready_to_depart=0,
            idx_deadlock_trains=[],
            children={},
        )
        graph_observation = Vertex(depth=0, node=root_node_observation, edges=[])

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)

        # track visited cells.
        visited = OrderedSet()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        orientation = agent.direction
        branch_directions = [(orientation + i) % 4 for i in range(-1, 3)]
        for i, branch_direction in enumerate(branch_directions):
            direction = self.tree_explored_actions_char[i]
            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_virtual_position, branch_direction)
                self._nodes_expanding.append((graph_observation, direction))
                branch_observation, branch_visited = self._explore_branch(handle, new_cell, branch_direction, 1, 1)
                root_node_observation.children[direction] = branch_observation
                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                graph_observation.edges.append(-np.inf)
                root_node_observation.children[direction] = -np.inf
        self.env.dev_obs_dict[handle] = visited
        assert len(self._nodes_expanding) == 0, 'Nodes left to expand.'
        return root_node_observation, graph_observation

    @staticmethod
    def _identify_crossing(transition_bit: str):
        """Given a string of bits returns true if this is equal to the crossing bits.

        :param transition_bit: string of bits.
        :return: boolean set to true if the input param matches the expected <1000010000100001>
        """
        return int(transition_bit, 2) == int('1000010000100001', 2)

    # pylint: disable=too-many-nested-blocks
    def _explore_tree_direction(  # noqa: max-complexity: 36
        self,
        agent: flatland.EnvAgent,
        projected_position: tuple[int, int],
        distance_map_handle: np.ndarray,
        agent_id: int,
        projected_direction: int,
        tot_dist: int,
    ) -> tuple[CellType, OrderedSet, Node, tuple[int, int], int, int]:
        """Helper method to project the agent forward onto the future path.

        :param agent: Agent to be projected in the exploration.
        :param projected_position: The projected position of the agent
        :param distance_map_handle: The distance map of the current agent.
        :param agent_id: The id of the agent
        :param projected_direction: The projected direction of the agent.
        :param tot_dist: the total distance from the agent position to the analysed cell.
        :return: 6-element tuple: #1 type of the cell, #2 set of visited cells, #3 node explored,
                                  #4 last projected position, #5 last direction of the agent,  #6 total distance.
        """

        # time required to travel a single cell.
        time_per_cell = 1.0 / agent.speed_counter.speed
        # feature tracking to fill the node's param.
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        distance_of_potential_conflict = np.inf
        conflicting_agent = -1
        position_of_conflict = (-1, -1)
        ca_distance_to_conflict = np.inf
        ca_has_alternative = None
        ca_expected_delay_for_alternative_path = -1
        conflict_on_junction_with_multiple_path = None
        expected_delay_for_2nd_best_path = -1
        unusable_switch = np.inf
        unusable_switch_cell = None
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.0
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0
        idx_trains_encountered_deadlocked = []
        last_cell_type: CellType = CellType.UNDEFINED
        visited = OrderedSet()
        exploring = True

        while exploring:
            if projected_position in self.location_has_agent_deadlocked:
                idx_trains_encountered_deadlocked.append(self.location_has_agent_deadlocked.get(projected_position))
            if self.location_has_agent.get(projected_position, 0) == 1:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[projected_position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[projected_position]

                other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(
                    projected_position, 0
                )

                if self.location_has_agent_direction[projected_position] == projected_direction:
                    # Cumulate the number of agents on branch with same direction
                    other_agent_same_direction += 1

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[projected_position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

                else:
                    # If no agent in the same direction was found all agents in that position are other direction
                    # Attention this counts to many agents as a few might be going off on a switch.
                    other_agent_opposite_direction += self.location_has_agent[projected_position]

                # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*projected_position, projected_direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*projected_position))
            total_transitions = transition_bit.count('1')
            crossing_found = self._identify_crossing(transition_bit)

            # Register possible future conflict
            predicted_time = int(tot_dist * time_per_cell)
            # if predicted_time == max_prediction_depth then we do not have data to do the projection.
            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [projected_position])[0]
                assert isinstance(int_position, int)
                assert 0 < tot_dist < self.max_prediction_depth
                pre_step = max(0, predicted_time - 1)
                post_step = min(self.max_prediction_depth - 1, predicted_time + 1)
                for time in (predicted_time, pre_step, post_step):
                    if int_position in fast_delete(self.predicted_pos[time], agent.handle):
                        conflict_found, conflict_info = self.check_potential_conflict(
                            time,
                            int_position,
                            agent,
                            distance_of_potential_conflict,
                            tot_dist,
                            projected_direction,
                            cell_transitions,
                            predicted_time,
                        )
                        if conflict_found:
                            (
                                conflicting_agent,
                                position_of_conflict,
                                ca_distance_to_conflict,
                                ca_has_alternative,
                                ca_expected_delay_for_alternative_path,
                                conflict_on_junction_with_multiple_path,
                                expected_delay_for_2nd_best_path,
                            ) = conflict_info
                            distance_of_potential_conflict = tot_dist
                            break  # we can stop the conflict search.

            if projected_position in self.location_has_target and projected_position != agent.target:
                if tot_dist < other_target_encountered:
                    other_target_encountered = tot_dist

            if projected_position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist

            if (projected_position[0], projected_position[1], projected_direction) in visited:
                last_cell_type = CellType.TERMINAL
                break
            visited.add((projected_position[0], projected_position[1], projected_direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if fast_position_equal(projected_position, self.env.agents[agent_id].target):
                last_cell_type = CellType.TARGET
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = fast_count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist
                unusable_switch_cell = projected_position

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_cell_type = CellType.DEAD_END
                else:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    projected_direction = fast_argmax(cell_transitions)
                    projected_position = get_new_position(projected_position, projected_direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_cell_type = CellType.SWITCH
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as already visited / not crossable
                print(
                    'WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell',
                    projected_position[0],
                    projected_position[1],
                    projected_direction,
                )
                last_cell_type = CellType.TERMINAL
                break
        # ends of exploration.

        if last_cell_type == CellType.TARGET:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_cell_type == CellType.TERMINAL:
            dist_to_next_branch = np.inf
            dist_min_to_target = distance_map_handle[projected_position[0], projected_position[1], projected_direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = distance_map_handle[projected_position[0], projected_position[1], projected_direction]

        ca_can_use_the_switch = -1
        delay_for_using_the_switch = -1
        # check whether the conflicting_agent could use the switch. Need to force the check as the switch may be closer
        # than the conflicting cell.
        if conflicting_agent != -1 and not np.isinf(unusable_switch):
            short_path_pred = self.predictions[conflicting_agent]
            idx_switch = np.where(
                (short_path_pred[:, 1] == unusable_switch_cell[0]) & (short_path_pred[:, 2] == unusable_switch_cell[1])
            )[0]
            if len(idx_switch) > 0:
                dir_conflicting_agent = short_path_pred[idx_switch, 3][0]
                can_use_the_switch, delay_to_use_the_switch = self.exist_alternative_path_on_switch(
                    unusable_switch_cell, dir_conflicting_agent, conflicting_agent, projected_direction
                )

                if can_use_the_switch:
                    ca_can_use_the_switch = True
                    delay_for_using_the_switch = delay_to_use_the_switch
                else:
                    ca_can_use_the_switch = False
        node = Node(
            dist_own_target_encountered=own_target_encountered,
            dist_other_target_encountered=other_target_encountered,
            dist_other_agent_encountered=other_agent_encountered,
            dist_potential_conflict=distance_of_potential_conflict,
            idx_conflicting_agent=conflicting_agent,
            cell_of_conflict=position_of_conflict,
            dist_other_agent_to_conflict=ca_distance_to_conflict,
            dist_unusable_switch=unusable_switch,
            unusable_switch_usable_for_ca=ca_can_use_the_switch,
            unusable_switch_delay_for_ca=delay_for_using_the_switch,
            dist_to_next_branch=dist_to_next_branch,
            clashing_agent_has_alternative=ca_has_alternative,
            ca_expected_delay_for_alternative_path=ca_expected_delay_for_alternative_path,
            conflict_on_junction_with_multiple_path=conflict_on_junction_with_multiple_path,
            expected_delay_for_2nd_best_path=expected_delay_for_2nd_best_path,
            dist_min_to_target=dist_min_to_target,
            num_agents_same_direction=other_agent_same_direction,
            num_agents_opposite_direction=other_agent_opposite_direction,
            num_agents_malfunctioning=malfunctioning_agent,
            speed_min_fractional=min_fractional_speed,
            num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
            idx_deadlock_trains=idx_trains_encountered_deadlocked,
            children={},
        )
        return last_cell_type, visited, node, projected_position, projected_direction, tot_dist

    def _explore_branch(
        self, agent_id: int, position: tuple[int, int], direction: int, tot_dist: int, depth: int
    ) -> tuple[Node, OrderedSet]:
        """
        Recursive utility function to compute both tree-based and graph based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.

        :param agent_id: the current agent id.
        :param position: the projected position of the agent.
        :param direction: the projected direction of the agent.
        :param tot_dist: the total number of cells that have to be crossed to get to the projected cell.
        :param depth: the depth of the exploration for the graph obs builder built along the tree obs.
        :return: root of the tree-based observation (node) and visited cells along with the direction of visit.
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            # A branch may exist, but we don't look beyond that.
            # We return the empty instead of setting the direction to -np.inf
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops

        agent = self.env.agents[agent_id]
        distance_map_handle = self.env.distance_map.get()[agent_id]

        # iterate through the path...
        (last_cell_type, visited, node, position, direction, tot_dist) = self._explore_tree_direction(
            agent, position, distance_map_handle, agent_id, direction, tot_dist
        )

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)

        # remove last node enqueed and create new vertex
        _node_expanded, _direction_expanded = self._nodes_expanding.pop()
        current_vertex = Vertex(depth=depth, node=node, edges=[])
        _node_expanded.edges.append(current_vertex)

        # recursive exploration.
        branch_directions = [(direction + 4 + i) % 4 for i in range(-1, 3)]
        for i, branch_direction in enumerate(branch_directions):
            _direction = self.tree_explored_actions_char[i]
            if last_cell_type == CellType.DEAD_END and self.env.rail.get_transition(
                (*position, direction), (branch_direction + 2) % 4
            ):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = get_new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(
                    agent_id, new_cell, (branch_direction + 2) % 4, tot_dist + 1, depth + 1
                )
                node.children[_direction] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited

            elif last_cell_type == CellType.SWITCH and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)

                # if a switch is found then a new junction-node is expanded.
                if depth < self.max_depth:
                    self._nodes_expanding.append((current_vertex, _direction))

                branch_observation, branch_visited = self._explore_branch(
                    agent_id, new_cell, branch_direction, tot_dist + 1, depth + 1
                )
                node.children[_direction] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            else:
                # no exploring possible, add just cells with infinity
                current_vertex.edges.append(-np.inf)
                node.children[_direction] = -np.inf

        if depth == self.max_depth:
            node.children.clear()
        return node, visited

    def set_env(self, env: Environment):
        """
        Set the environment of the observation to the one given.
        :param env: The environment to be set to the current instance.
        """

        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    @classmethod
    def _reverse_dir(cls, direction: int) -> int:
        """Return the reverse of a direction.
            The directions are from 0 to 3 clockwise with 0 being upward.
            So, opposite(0) -> 2 and so on.

        :param direction: current direction.
        :return: the opposite of the current direction
        """
        return int((direction + 2) % 4)

    def check_potential_conflict(
        self,
        predicted_time: int,
        int_position: int,
        agent: EnvAgent,
        closest_conflict: float,
        distance_to_projected_cell: float,
        projected_direction: int,
        cell_transitions: tuple[int, int, int, int],
        reference_time: int,
    ) -> tuple[bool, None | tuple[int, tuple[int, int], float, None | bool, int, bool, int]]:
        """Method to look for potential conflict at a certain time.

        :param predicted_time: Time to check for conflict.
        :param int_position: Cell of conflict encoded as an integer.
        :param agent: Agent that is looking for conflicts.
        :param closest_conflict: Distance between the agent and the closest conflict already identified.
        :param distance_to_projected_cell: Distance, in cells, between the agent and the potential conflict.
        :param projected_direction: Direction of the agent projected into the future.
        :param cell_transitions: Transition possible from the cell that will be occupied at the conflict time.
        :param reference_time: Reference time for the projection. Needed to define which train will
                                occupy the cell of interest in case of a conflict.
        :return: Tuple with boolean value flagging whether an interesting conflict is found.
                 A conflict is interesting if the distance of the found conflict is lower
                 than the distance of the already identified conflicts.
                 If the flag is set to true, then it returns:
                    - handle of the conflicting train;
                    - cell of conflict, as a tuple;
                    - distance to conflict for the other train;
                    - boolean value representing whether the other train has an alternative path that could follow.
                    - Integer value with the expected delay for the alternative path of the conflicting agent.
                    - boolean value representing whether it exists another path for the current agent.
                    - Integer value with the expected delay for the alternative path for the current agent.

        """
        potential_conflict_identified = False
        position_of_conflict = None
        conflicting_agent = None
        conflicting_train_distance_to_conflict = None
        conflicting_agent_has_alternative = None
        ca_delay_for_alternative = -1
        predicted_pos = np.asarray(self.predicted_pos[predicted_time])
        conflicting_agents = fast_where(predicted_pos == int_position)

        exist_alternative_path = False
        delay_for_alternative_path = -1

        for ca_handle in conflicting_agents:
            ca_predicted_dir = self.predicted_dir[predicted_time][ca_handle]
            if (
                projected_direction != ca_predicted_dir
                and cell_transitions[self._reverse_dir(ca_predicted_dir)] == 1
                and distance_to_projected_cell < closest_conflict
            ):
                conflicting_agent = self.env.agents[ca_handle]
                position_of_conflict = position_to_coordinate(self.env.width, [int_position])[0]
                conflicting_train_distance_to_conflict = self.distance_on_rail(conflicting_agent, position_of_conflict)
                delta_time = predicted_time - reference_time
                conflicting_agent_has_alternative, ca_delay_for_alternative = self.has_alternative_path(
                    train_handle=ca_handle,
                    conflict_pos=position_of_conflict,
                    conflicting_handle=agent.handle,
                    delta_time=delta_time,
                )
                potential_conflict_identified = True

                exist_alternative_path, delay_for_alternative_path = self.exist_2nd_best_path(
                    position_of_conflict, projected_direction, agent.handle
                )
                # if we found a conflict already -> break.
                break
        # if agent is ready to depart we must evaluate the case. Otherwise, if the ca can resolve the conflict
        # then we disregard it and look beyond it.
        if (
            agent.state != TrainState.READY_TO_DEPART
            and conflicting_train_distance_to_conflict == 0
            and distance_to_projected_cell == 1
        ):
            if conflicting_agent.position == position_of_conflict:
                # if not can move/path not possible:
                if conflicting_agent_has_alternative:
                    potential_conflict_identified = False
        if not potential_conflict_identified:
            return potential_conflict_identified, None
        return (
            potential_conflict_identified,
            (
                conflicting_agent.handle,
                position_of_conflict,
                conflicting_train_distance_to_conflict,
                conflicting_agent_has_alternative,
                ca_delay_for_alternative,
                exist_alternative_path,
                delay_for_alternative_path,
            ),
        )

    def has_alternative_path(
        self, train_handle: int, conflict_pos: tuple[int, int], conflicting_handle: int, delta_time: int
    ) -> tuple[None | bool, int]:
        """Looks into future position to check whether a train has an alternative path to the target.

        :param train_handle: Idx of the train to look for alternative paths.
        :param conflict_pos: Position of the conflict as coordinates.
        :param conflicting_handle: Handle of the conflicting train.
        :param delta_time: Delta of time between the reference time and the prediction time.
                            This is needed to establish who will occupy the cell first.
                            - 1 -> looking after the step, meaning conflicting_handle will take the cell and
                                    train handle cannot evaluate the path.
        :return: tuple with boolean flag and the expected delay, boolean flag can be:
                    - None if it is not possible to establish whether exist or not an alternative path.
                    - True if it exists a valid alternative path.
                    - False if it does not exist / agent already on the path
        """
        # if agent not yet on track then always possible to delay the start by 1 timestep.
        agent = self.env.agents[train_handle]
        if agent.state.is_off_map_state():
            target_distance = _fetch_geodesic_distance(
                train_handle,
                agent.initial_position,
                agent.direction,
                self.env.distance_map,
            )
            # +1 is the expected delay to stay put for a timestep.
            travel_time = (target_distance / agent.speed_counter.speed) + 1
            arrival_delay = max(0, (travel_time + self.env._elapsed_steps) - agent.latest_arrival)
            if np.isinf(arrival_delay):
                # agent will not depart,
                # so we could safely assume that being stopped for 1 timestep does not make any difference.
                arrival_delay = 0
            return True, arrival_delay

        shortest_path = self.predictions[train_handle]
        must_return = False
        upper_index = len(shortest_path)
        nan_mask = np.isnan(shortest_path)
        if True in nan_mask:
            upper_index = np.where(nan_mask)[0][0]
        # if not cast to integer then triggers IndexError from time to time.
        # The issue could be related to the lru caching.

        for directed_pos in shortest_path[:upper_index].astype(int):
            if must_return:
                return False, -1
            if tuple(directed_pos[1:3]) == conflict_pos:
                # The cell will be taken by the conflicting train.
                if delta_time > 0:
                    return False, -1
                # when delta time is 0 then we need to look at the cell of conflict only when
                # the train_handle moves before the conflicting_handle.
                if delta_time == 0 and train_handle > conflicting_handle:
                    return False, -1
                # Allow to explore the cell of the conflict.
                must_return = True
            cell = tuple(directed_pos[1:3])
            direction = directed_pos[3]
            exist_alternative, delay_for_alternative = self.exist_2nd_best_path(cell, direction, train_handle)
            if exist_alternative:
                # this does not guarantee that we find the 'best' alternative path.
                return exist_alternative, delay_for_alternative
        return None, -1

    def exist_2nd_best_path(self, cell: tuple[int, int], direction: int, train_handle: int) -> tuple[bool, None | int]:
        """Given a cell, a train handle and a direction it returns whether the train has an alternative path
            to the shortest one and an optimistic estimation for the expected delay.

        :param cell: Cell from which estimate the future possible paths.
        :param direction: Direction of the agent at the given cell.
        :param train_handle: Handle of the train for the estimation.
        :return: Tuple with boolean flag and expected delay.
        """
        possible_transitions = self.env.rail.get_transitions(*cell, direction)
        agent = self.env.agents[train_handle]
        if sum(possible_transitions) > 1:
            time_left = self.env._max_episode_steps - self.env._elapsed_steps
            arrival_time_paths = []
            for new_direction, possible in enumerate(possible_transitions):
                if not possible:
                    continue
                new_pos = get_new_position(cell, new_direction)
                distance_to_target = _fetch_geodesic_distance(
                    train_handle, new_pos, new_direction, self.env.distance_map
                )
                time_to_target = distance_to_target / agent.speed_counter.speed
                if time_to_target <= time_left:
                    # we allow for a possible path whenever the train can reach its target destination
                    # before the end of the episode.
                    # This could consider the delay as well, if we were to consider each possible path.
                    arrival_delay = max(0, (time_to_target + self.env._elapsed_steps) - agent.latest_arrival)
                    arrival_time_paths.append(arrival_delay)
            if len(arrival_time_paths) > 1:
                return True, sorted(arrival_time_paths)[1]
        return False, -1

    def exist_alternative_path_on_switch(
        self, cell: tuple[int, int], direction: int, train_handle: int, direction_incoming_agent: int
    ) -> tuple[bool, None | int]:
        """Given a cell, a train handle and a direction it returns whether the train can use a switch
            along with the expected delay.

        :param cell: Cell from which estimate the future possible paths.
        :param direction: Direction of the agent at the given cell.
        :param train_handle: Handle of the train for the estimation.
        :param direction_incoming_agent: In case of a switch, we must consider the direction blocked by the other train.
        :return: Tuple with boolean flag and expected delay.
        """
        agent = self.env.agents[train_handle]
        possible_transitions = list(self.env.rail.get_transitions(*cell, direction))
        # get the complementary of the direction given that the other agent is following that direction
        complementary_dir = (direction_incoming_agent + 2) % 4
        # set the complementary direction as not usable.
        possible_transitions[complementary_dir] = 0
        if sum(possible_transitions) > 0:
            time_left = self.env._max_episode_steps - self.env._elapsed_steps
            arrival_time_paths = []
            for new_direction, possible in enumerate(possible_transitions):
                if not possible:
                    continue
                new_pos = get_new_position(cell, new_direction)
                distance_to_target = _fetch_geodesic_distance(
                    train_handle, new_pos, new_direction, self.env.distance_map
                )
                time_to_target = distance_to_target / agent.speed_counter.speed
                if time_to_target <= time_left:
                    # we allow for a possible path whenever the train can reach its target destination
                    # before the end of the episode.
                    # This could consider the delay as well, if we were to consider each possible path.
                    arrival_delay = max(0, (time_to_target + self.env._elapsed_steps) - agent.latest_arrival)
                    arrival_time_paths.append(arrival_delay)
            if len(arrival_time_paths) > 0:
                return True, sorted(arrival_time_paths)[0]
        return False, -1

    def distance_on_rail(self, agent: EnvAgent, target_cell: tuple[int, int]) -> int:
        """Given an agent and a target cell, it returns the distance between the agent position and the cell.
            In the rare case that it is not possible to measure the exact distance given the rail's config, the distance
            is estimated via Manhattan distance formula.

        :param agent: Agent to consider for the distance measurement.
        :param target_cell: Target cell to consider for the distance measurement.
        :return: Distance between the current position of the agent given and the target cell specified.
        """
        # lookup on the shortest path to check the distance to the cell of conflict
        distance_from_short_path = np.where(np.all(self.predictions[agent.handle][:, 1:3] == target_cell, 1))[0]
        if len(distance_from_short_path) == 1:
            return distance_from_short_path[0]

        # if not found then return Manhattan distance
        return distance_on_rail(_get_agent_position(agent), target_cell, 'Manhattan')
