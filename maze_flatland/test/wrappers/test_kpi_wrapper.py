"""File holding the tests for the KPIs wrapper."""
from __future__ import annotations

from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.test.env_instantation import create_env_for_testing
from maze_flatland.wrappers.kpis_events_wrapper import KpisEventsWrapper


def test_wrapper():
    """Runs a rollout and checks the output of the KPIs wrapper."""
    env = create_env_for_testing()
    env = KpisEventsWrapper.wrap(env)
    greedy_policy = GreedyPolicy()
    done = False
    info = {}
    env.seed(1234)
    obs = env.reset()
    while not done:
        action = greedy_policy.compute_action(obs, env.get_maze_state(), env, env.actor_id())
        obs, _, done, info = env.step(action)
    ms = env.get_maze_state()
    assert 'kpis' in info, 'KPIs not stored in the info dict.'
    assert 'events' in info, 'Events not stored in the info dict.'
    dead_trains = [train.deadlock for train in ms.trains]
    if any(dead_trains):
        assert info['kpis']['Deadlock ratio'] == dead_trains.count(True) / len(dead_trains)
        for idx, is_dead in enumerate(dead_trains):
            pos_train = ms.trains[idx].position
            assert f'[Deadlock] Train {idx} at position {pos_train}' in info['events']
    success_rate = [train.is_done() for train in ms.trains]
    assert info['kpis']['Arrival ratio'] == success_rate.count(True) / len(success_rate)
