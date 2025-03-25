from __future__ import annotations

import os.path
import shutil
import time

import numpy as np
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.minimal import MinimalObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper


# pylint: disable=c-extension-no-member
def _create_example_env(n_trains: int, malfunction_rate: float, n_cities: int, speed_map: dict[float, float]):
    """Create a small example env."""
    map_width = 30
    map_height = 30
    core_env = create_core_env(n_trains, map_width, map_height, n_cities, malfunction_rate, speed_map, False)
    env = FlatlandEnvironment(
        core_env,
        {'train_move': DirectionalAC()},
        {'train_move': MinimalObservationConversion(False)},
    )
    return FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())


def render_random_rollout():
    """Render a random rollout and save the env stat renderings to the disk"""
    speed_map = {1: 0.5, 0.5: 0.5}
    env = _create_example_env(1, 0, 2, speed_map)
    env.seed(1967693548)
    rng = np.random.RandomState(1235)

    out_path = os.path.abspath('./env_rendering')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    start_time = time.time()
    times = []
    obs = env.reset()

    actions = []
    action_masks = []
    print(f'output dir: {out_path}')
    done = False
    while not done:
        state: FlatlandMazeState = env.get_maze_state()
        action = rng.choice(np.where(obs['train_move_mask'])[0])

        if state.current_train_id == 0:
            print(f'Timestep: {state.env_time} - {state.current_train_id}:')
        train = state.trains[state.current_train_id]
        print(
            f'\t train id: {train.handle}' f'\n\t direction: {train.direction}',
            f'\n\t status: {train.status},',
            f'\n\t speed: {train.speed}',
            f'\n\t in_transition: {train.in_transition}',
            f'\n\t position: {train.position}',
            f'\n\t target distance: {train.target_distance}',
            f'\n\t mask: {obs["train_move_mask"]}',
            f'\n\t taking action: {action}',
        )

        times.append(time.time() - start_time)

        actions.append(action)
        action_masks.append(obs['train_move_mask'])
        if len(actions) == state.n_trains:
            env.render(action=actions, save_in_dir=out_path)
            actions = []
            action_masks = []
        start_time = time.time()
        obs, rew, done, info = env.step({'train_move': action})


if __name__ == '__main__':
    render_random_rollout()
