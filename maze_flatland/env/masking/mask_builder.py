"""File holding the masking classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from flatland.core.grid.grid4_utils import MOVEMENT_ARRAY
from flatland.envs.step_utils.states import TrainState
from maze_flatland.env.backend_utils import (
    CELL_TYPE_SWITCH,
    get_cell_int_and_type,
    get_cell_int_for_pos,
    identify_cell_type,
)
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import MazeTrainState


class TrainLogicMask:
    """Class holding the positional masking condition specific for a train.

    :param handle: Integer ID of the train.
    :param skip_decision: Boolean flag to skip the decision step.
    :param in_transition: Boolean flag to indicate that the agent is transitioning between two cells.
    :param possible_next_positions: List of tuple containing the possible next positions.
    :param info: Additional information about the current mask.
    """

    def __init__(
        self,
        handle: int,
        skip_decision: bool,
        in_transition: bool,
        possible_next_positions: any,
        info: str | None = None,
    ):
        self.train_handle = handle

        # Whether the decision should be skipped.
        self.skip_decision = skip_decision
        # Whether the train is transitioning between two cells.
        self.in_transition = in_transition
        # Whether stop should be included as an option.
        self.possible_next_positions = possible_next_positions
        self._info = info

    def explain(self) -> str:
        """Provides explanation for the masking (if available).

        :return: A string with the explanation of the masking."""
        if self._info is None:
            return 'Explanation not available.'
        return self._info

    def only_single_option(self) -> bool:
        """Based on the conditions, returns True if there is only a single choice in the pool of available options."""
        if self.skip_decision or self.in_transition or len(self.possible_next_positions) == 1:
            return True
        return False


class LogicMaskBuilderInterface(ABC):
    """Defines the interface class for the logic to build the mask."""

    @abstractmethod
    def create_train_mask(self, train_state: MazeTrainState, transition_map: np.ndarray) -> TrainLogicMask:
        """Create a mask based on the train status and the rail configuration.

        :param train_state: MazeTrainState object containing the state for a train.
        :param transition_map: Rail configuration as a transition map.

        :return: TrainLogicMask object containing the logical mask.
        """

        raise NotImplementedError()


class LogicMaskBuilder(LogicMaskBuilderInterface):
    """Defines the mask builder class to mask out impossible decisions for an agent.

    :param disable_stop_on_switches: [Default = False] Boolean value to prevent trains for stopping on a junction.
    :param mask_out_dead_ends: [Default = True] Whether to mask out dead ends from the masking.
    """

    def __init__(self, disable_stop_on_switches: bool = False, mask_out_dead_ends: bool = True):
        self.mask_out_dead_ends = mask_out_dead_ends
        self.disable_stop_on_switches = disable_stop_on_switches

    def create_train_mask(self, train_state: MazeTrainState, transition_map: np.ndarray) -> TrainLogicMask:
        """Creates a mask based on the current state of a train and the rail configuration.
            It determines the possible next positions and options for a train.
            Logic steps:
            1. If the train is in transition, it is flagged accordingly.
            2. If the train is waiting, malfunctioning, or finished, flag to skip decision.
            3. If the train is in a deadlock, its position is returned as the only option.
            4. Otherwise, possible next positions are extracted from the train's action state.
            5. After evaluating stopping conditions, current train position may be added to the possible next positions.
            6. If dead-end masking is enabled,
                directions leading to a path not connected to the train's target are removed.

        :param train_state: The current MazeTrainState for the given train.
        :param transition_map: The transition map of the rail detailing its configuration.

        :return: TrainMask object containing the mask.
        """

        in_transition = False
        skip_decision = False
        possible_next_positions = []
        info = None
        if train_state.in_transition:
            in_transition = True
            info = '\t\tTrain is transitioning.'
        elif train_state.status in [TrainState.WAITING, TrainState.MALFUNCTION_OFF_MAP]:
            # In case the train is out of map and not ready to depart.
            skip_decision = True
            info = '\t\tTrain is waiting or malfunctioning outside of map.'
        elif train_state.is_done():
            # In case the train is done already
            skip_decision = True
            info = '\t\tTrain is done already.'
        elif train_state.deadlock:
            # In case the train is dead
            possible_next_positions = [train_state.position]
            info = '\t\tTrain is in a deadlock.'
        if info is not None:
            return TrainLogicMask(train_state.handle, skip_decision, in_transition, possible_next_positions, info)

        # get possible next positions excluding the current one
        possible_next_positions = list(
            action_state.target_cell
            for action_state in train_state.actions_state.values()
            if action_state.target_cell is not None
        )
        info = f'\t\t[+] {len(possible_next_positions)} available directions.\n'
        # If needed, append the current position.
        flag_stop, reason_for_stop = self._check_if_stop_action_should_be_added(
            train_state,
            transition_map,
            possible_next_positions,
            self.disable_stop_on_switches,
        )
        if flag_stop:
            possible_next_positions.append(train_state.position)
            info += f'\t\t[+] Stop action is enabled ({reason_for_stop}).'
        else:
            info += '\t\t[-] Stop action is disabled.'
        # Remove all directions that lead to a dead end we can not recover from.
        if self.mask_out_dead_ends:
            n_pos_before_removing_dead_ends = len(possible_next_positions)
            possible_next_positions = self._remove_dead_end_directions(
                train_state,
                possible_next_positions,
            )
            delta_pos = n_pos_before_removing_dead_ends - len(possible_next_positions)
            if delta_pos > 0:
                info += f'\n\t\t[-] Removed {delta_pos} directions as considered as dead ends.'

        return TrainLogicMask(train_state.handle, skip_decision, in_transition, possible_next_positions, info)

    @classmethod
    def _check_if_stop_action_should_be_added(
        cls,
        train: MazeTrainState,
        transition_map: np.ndarray,
        possible_next_positions: list[tuple[int, int]],
        disable_stop_on_junction: bool,
    ) -> tuple[bool, str]:
        """Return true if a stop action should be added as a valid option to the mask.

        A stop action is valid if the train is located in a cell immediately before
        a diamond crossing or a switch where two paths merge into a single track.
        If disable_stop_on_switches is set to true this is only allowed if the current cell is not a switch.

        :param train: MazeTrainState of the train
        :param transition_map: Transition map for the train.
        :param possible_next_positions: The possible next train positions.
        :param disable_stop_on_junction: Whether to disable the stop action on a junction.

        :return: Tuple [bool, str] indicating if the stop action should be added and why.
        """
        assert len(possible_next_positions) > 0, 'No valid position found'

        # In case the agent is ready to depart always give the stop action as an option
        if train.status == TrainState.READY_TO_DEPART:
            return True, 'Train is ready to depart'

        # In case the disable stop on junction flag is set and the current cell type is a junction return false.
        cell_name, cell_type, cell_rotation = get_cell_int_and_type(transition_map, train.position)[1]
        if disable_stop_on_junction and cell_type in CELL_TYPE_SWITCH:
            return False, ''

        # Check if the next
        for x_pos, y_pos in possible_next_positions:
            p_cell_name, p_cell_type, p_cell_rotation = identify_cell_type(
                get_cell_int_for_pos(x_pos, y_pos, transition_map)
            )

            # Calculate the resulting direction when moving to the possible position.
            resulting_direction = MOVEMENT_ARRAY.index(tuple(np.array([x_pos, y_pos]) - train.position))

            if p_cell_type in [2, 10] and p_cell_rotation == resulting_direction:
                # in case the agent is facing a simple switch stopping does not bring any benefit in terms of letting
                # other trains pass.
                continue

            if p_cell_type == 6 and p_cell_rotation == resulting_direction:
                # in case the agent is facing a symmetric switch stopping does not bring any benefit in terms of letting
                # other trains pass.
                continue

            if p_cell_type in CELL_TYPE_SWITCH:
                return True, 'Train standing in front of a switch'
        return False, ''

    @classmethod
    def _remove_dead_end_directions(
        cls,
        train: MazeTrainState,
        possible_next_positions: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Check if any possible direction leads to dead - end (a place the train can never reach its target
        destination)

        :param train: MazeTrainState of the train
        :param possible_next_positions: The possible next positions.

        :return: The possible next positions where dead end states have been removed.
        """
        assert len(possible_next_positions) >= 1
        dead_end_positions = [
            FlatlandMazeAction.action_to_global_position(train.position, dead_action, train.direction)
            for dead_action in train.dead_ends
        ]
        if len(dead_end_positions) > 0:
            possible_next_positions = list(set(possible_next_positions) - set(dead_end_positions))
        return possible_next_positions
