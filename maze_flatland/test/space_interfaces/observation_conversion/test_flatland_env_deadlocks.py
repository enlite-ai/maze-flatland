"""
Tests deadlock-related functionality in FlatlandEnvironment.
"""

from __future__ import annotations

from flatland.envs.agent_utils import TrainState
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import _node_visited_status, cycling_block_identifier, detect_blocks_and_obstructions
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion
from maze_flatland.test.env_instantation import create_core_env


def _generate_test_env() -> FlatlandEnvironment:
    """
    Generates fixed test environment for deadlock test.
    """

    core_env: FlatlandCoreEnvironment = create_core_env(3, 30, 30, 2, 0, {1.0: 0.7, 1.0 / 2.0: 0.3})
    env = FlatlandEnvironment(
        core_env,
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': PositionalObservationConversion(False)},
    )
    env.seed(2)
    env.reset()

    return env


def step_all(env, action_0: int, action_1: int, action_2: int) -> None:
    """Steps all three agents."""
    env.step({'train_move': action_0})
    env.step({'train_move': action_1})
    env.step({'train_move': action_2})


def test_deadlock():
    """Tests whether a deadlock is correctly identified."""
    env = _generate_test_env()
    while TrainState.READY_TO_DEPART not in [train.status for train in env.get_maze_state().trains]:
        step_all(env, 0, 0, 0)
    step_all(env, 0, 0, 2)  # 3rd train departure
    step_all(env, 0, 0, 4)  # 3rd train pause
    while TrainState.READY_TO_DEPART not in [train.status for train in env.get_maze_state().trains]:
        step_all(env, 0, 0, 4)

    step_all(env, 2, 0, 4)
    step_all(env, 4, 2, 4)
    step_all(env, 4, 4, 4)
    _ = [step_all(env, 2, 2, 4) for _ in range(2)]
    _ = [step_all(env, 3, 2, 2) for _ in range(3)]
    step_all(env, 4, 2, 2)
    step_all(env, 4, 2, 4)
    step_all(env, 4, 1, 4)
    step_all(env, 4, 3, 4)
    step_all(env, 4, 2, 4)
    step_all(env, 2, 2, 2)
    _ = [step_all(env, 4, 2, 2) for _ in range(2)]

    blocks = env.observation_conversion.maze_to_space(env.get_maze_state())['train_blocks']
    assert (blocks == [[0, 1, 0], [1, 0, 0], [1, 0, 0]]).all()

    # assert env.get_maze_state().deadlocks == {0, 1, 2}


def test_flatland_no_deadlocks():
    """
    Tests whether 1) a follow-up situation, in which a train is one cell behind an advancing train,
    is correctly identified as not a deadlock; 2) two train facing each other are not in deadlock as
    a turn is a valid move.
    """

    env = _generate_test_env()
    while TrainState.READY_TO_DEPART not in [train.status for train in env.get_maze_state().trains]:
        step_all(env, 0, 0, 0)
    step_all(env, 0, 0, 2)  # 3rd train departure
    step_all(env, 0, 0, 4)  # 3rd train pause
    while TrainState.READY_TO_DEPART not in [train.status for train in env.get_maze_state().trains]:
        step_all(env, 0, 0, 4)

    step_all(env, 2, 0, 4)
    step_all(env, 4, 2, 4)
    step_all(env, 4, 4, 4)
    _ = [step_all(env, 2, 2, 4) for _ in range(2)]
    _ = [step_all(env, 3, 2, 2) for _ in range(3)]
    step_all(env, 4, 2, 2)
    step_all(env, 4, 2, 4)
    step_all(env, 4, 1, 4)
    step_all(env, 4, 3, 4)
    step_all(env, 4, 2, 4)
    step_all(env, 4, 2, 2)
    step_all(env, 4, 2, 2)
    step_all(env, 4, 2, 2)
    step_all(env, 4, 2, 4)
    step_all(env, 4, 4, 4)
    # assert env.get_maze_state().deadlocks == set()  # check no deadlock as train 1 can turn.
    blocks = env.observation_conversion.maze_to_space(env.get_maze_state())['train_blocks']
    assert (blocks == [[0, 1, 0], [0, 0, 0], [1, 0, 0]]).all()
    step_all(env, 3, 1, 3)
    blocks = env.observation_conversion.maze_to_space(env.get_maze_state())['train_blocks']

    assert (blocks == [[0, 0, 0], [0, 0, 0], [1, 0, 0]]).all()


def test_cycling_block_identified_ms():
    def run_dummy_cyclic_block_identifier(n_trains: int, node_relations: dict[int : list[int]]):
        deadlocks = []
        node_visited = [_node_visited_status.NOT_VISITED for _ in range(n_trains)]
        for tid in range(n_trains):
            if cycling_block_identifier(tid, node_visited, node_relations):
                deadlocks.append(tid)
        return deadlocks

    def simple_deadlock_case():
        """Simple test on cycling blocks."""
        n_trains = 3
        node_relations = {0: [1], 1: [2], 2: [0]}
        assert [0, 1, 2] == (run_dummy_cyclic_block_identifier(n_trains, node_relations))

    def simple_non_deadlock_case():
        """Simple test on cycling blocks.
        No deadlock as 2 could move as soon as 3 (which is not blocked) moves out.
        """
        n_trains = 4
        node_relations = {0: [1], 1: [2], 2: [0, 3]}
        assert not run_dummy_cyclic_block_identifier(n_trains, node_relations)

    def complex_case_with_deadlock():
        """Test complex case with deadlocks."""
        n_trains = 30
        expected_trains_dead = [0, 11, 13, 14, 21, 23, 25, 27, 28, 29]
        node_relations = {
            0: [27],
            2: [24],
            7: [2],
            11: [29, 23],
            13: [21],
            14: [29],
            19: [7],
            21: [23],
            23: [11],
            25: [13],
            27: [0, 21],
            28: [14],
            29: [11, 14],
        }
        trains_dead = run_dummy_cyclic_block_identifier(n_trains, node_relations)
        assert sorted(trains_dead) == sorted(expected_trains_dead)

    simple_deadlock_case()
    simple_non_deadlock_case()
    complex_case_with_deadlock()


# pylint: disable=too-many-statements
def test_blocks_based_on_status():
    """Produce more tests aligned with reality."""

    class dummy_action_state:
        """Dummy action state class as the one in ~.maze_state.MazeTrainState."""

        def __init__(self, target_cell: tuple[int, int], is_dead_end: bool):
            self.target_cell = target_cell
            self.obstructed_by = None
            self.dead_end = is_dead_end

        def update_block(self, t_idx: int):
            assert not self.obstructed
            self.obstructed_by = t_idx  #

        @property
        def obstructed(self):
            return self.obstructed_by is not None

        def is_safe(self):
            return not self.dead_end

    class DummyMazeTrainState:
        """Dummy maze train state class as the one in ~.maze_state.MazeTrainState."""

        def __init__(
            self,
            idx: int,
            position: tuple[int, int],
            status: TrainState,
            actions_state: dict[FlatlandMazeAction:dummy_action_state],
        ):
            self.handle = idx
            self.position = position
            self.status = status
            self.actions_state = actions_state

        def is_on_map(self):
            return self.status in [TrainState.MALFUNCTION, TrainState.STOPPED, TrainState.MOVING]

        def is_block(self):
            return all(
                action_state.obstructed for action_state in self.actions_state.values() if action_state.is_safe()
            )

    def test_trains_facing_each_other():
        """Simple test, 2 trains blocking each other."""
        n_trains = 2
        trains_pos = [(0, 0), (1, 1)]
        trains_status = [TrainState.MOVING, TrainState.MOVING]
        trains_actions_states = [
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(1, 1), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=True),
            },
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(0, 0), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=True),
            },
        ]
        train_states = [
            DummyMazeTrainState(idx, tp, ts, tas)
            for idx, tp, ts, tas in zip(range(n_trains), trains_pos, trains_status, trains_actions_states)
        ]
        _ = detect_blocks_and_obstructions(train_states)
        assert train_states[0].is_block()
        assert train_states[1].is_block()

    def test_trains_with_alternative_on_1():
        """Simple test, 2 trains blocking each other."""
        n_trains = 2
        trains_pos = [(0, 0), (1, 1)]
        trains_status = [TrainState.MOVING, TrainState.MOVING]
        trains_actions_states = [
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(1, 1), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=True),
            },
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(0, 0), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=False),
            },
        ]
        train_states = [
            DummyMazeTrainState(idx, tp, ts, tas)
            for idx, tp, ts, tas in zip(range(n_trains), trains_pos, trains_status, trains_actions_states)
        ]
        _ = detect_blocks_and_obstructions(train_states)
        assert train_states[0].is_block()
        assert not train_states[1].is_block()
        assert train_states[1].actions_state[FlatlandMazeAction.GO_FORWARD].obstructed
        assert not train_states[1].actions_state[FlatlandMazeAction.DEVIATE_LEFT].obstructed

    def test_ready_to_depart_train_blocked():
        """Simple test, a train would like to be placed on map but the position is already taken.
        As a result, the train will be blocked.
        """
        n_trains = 3
        trains_pos = [(0, 0), (0, 0), (1, 1)]
        trains_status = [TrainState.READY_TO_DEPART, TrainState.MOVING, TrainState.MOVING]
        trains_actions_states = [
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(1, 1), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=True),
            },
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(1, 0), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=False),
            },
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(0, 0), is_dead_end=False),
            },
        ]
        train_states = [
            DummyMazeTrainState(idx, tp, ts, tas)
            for idx, tp, ts, tas in zip(range(n_trains), trains_pos, trains_status, trains_actions_states)
        ]
        trains_blocked_by = detect_blocks_and_obstructions(train_states)
        assert train_states[0].is_block()
        assert 0 not in trains_blocked_by[2]

    def test_out_of_map_no_create_obstruction(ts0: TrainState, ts1: TrainState, ts2: TrainState):
        """Simple test, a train would like to be placed on map but the position is already taken.
        As a result, the train will be blocked.
        :param ts0: The state of the train with handle of 0.
        :param ts1: The state of the train with handle of 1.
        :param ts1: The state of the train with handle of 2.
        """
        n_trains = 3
        trains_pos = [(0, 0), (1, 0), (1, 1)]
        trains_status = [ts0, ts1, ts2]
        trains_actions_states = [
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(1, 1), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=True),
            },
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(1, 0), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 2), is_dead_end=False),
            },
            {
                FlatlandMazeAction.GO_FORWARD: dummy_action_state(target_cell=(0, 0), is_dead_end=False),
                FlatlandMazeAction.DEVIATE_LEFT: dummy_action_state(target_cell=(1, 0), is_dead_end=False),
            },
        ]
        train_states = [
            DummyMazeTrainState(idx, tp, ts, tas)
            for idx, tp, ts, tas in zip(range(n_trains), trains_pos, trains_status, trains_actions_states)
        ]
        trains_blocked_by = detect_blocks_and_obstructions(train_states)
        assert trains_blocked_by[2] == []
        n_trains_checked_for_blocks = len(
            [t for t in train_states if t.is_on_map() or t.status == TrainState.READY_TO_DEPART]
        )
        assert len(trains_blocked_by) == n_trains_checked_for_blocks

    # case both on map and blocking each other
    test_trains_facing_each_other()
    # both on map but trains 1 has alternative. 0->blocked.
    test_trains_with_alternative_on_1()
    # trains 0 ready to depart but blocked by trains 1.
    test_ready_to_depart_train_blocked()
    # no blocks as trains on target cells of 2 are not placed on the map..
    test_out_of_map_no_create_obstruction(TrainState.READY_TO_DEPART, TrainState.WAITING, TrainState.MOVING)
    test_out_of_map_no_create_obstruction(TrainState.MALFUNCTION_OFF_MAP, TrainState.DONE, TrainState.MALFUNCTION)
    test_out_of_map_no_create_obstruction(TrainState.READY_TO_DEPART, TrainState.WAITING, TrainState.STOPPED)
