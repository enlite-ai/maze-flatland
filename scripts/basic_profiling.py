"""Basic profiling of maze_env, core_env and rail_env."""

from __future__ import annotations

import time

import flatland
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from memory_profiler import profile


# pylint: disable=c-extension-no-member
def init_core_env(
    map_width,
    map_height,
    n_trains,
    n_cities,
    malfunction_rate,
    speed_ratio_map,
    include_maze_state_in_serialization=False,
) -> FlatlandCoreEnvironment:
    """Initialize an instance of the flatland environment.
    :param map_width: width of the gridmap
    :param map_height: height of the gridmap
    :param n_trains: number of trains within the system
    :param n_cities: number of cities to initialise on the rail
    :param malfunction_rate: probability that a train is subjected to a malfunction
    :param speed_ratio_map: list of speeds for the agents
    :param include_maze_state_in_serialization: whether to serialize the current state
    :return: an instance of the flatland core environment.
    """
    return create_core_env(
        n_trains,
        map_width,
        map_height,
        n_cities,
        malfunction_rate,
        speed_ratio_map,
        include_maze_state_in_serialization,
    )


def main_env_benchmark(n_iterations: int, n_env_steps: int, n_trains: list[int], map_sizes: list[int]) -> None:
    print('\n\nBasic Benchmarking...')
    print(f'Number of iterations: {n_iterations}')
    print(f'Number of environment steps: {n_env_steps}')
    print(f'Number of trains: [{", ".join(map(str, n_trains))}]')
    print(f'Map sizes: [{", ".join(map(str, map_sizes))}]\n\n')

    for map_size in map_sizes:
        print(f'Mapsize {map_size}\n')

        for n_train in n_trains:
            print(f'Trains {n_train}\n')

            # -- MAZE ENV --

            core_env = create_core_env(n_train, map_size, map_size, 3, 1 / 100, {1: 1})
            maze_env = FlatlandEnvironment(
                core_env=core_env,
                action_conversion={'train_move': DirectionalAC()},
                observation_conversion={'train_move': PositionalObservationConversion(True)},
            )

            # -- CORE ENV --

            env = init_core_env(map_size, map_size, n_train, 3, 1 / 100, speed_ratio_map={1: 1})

            # -- RAIL ENV --

            # pylint: disable=protected-access
            rail_env = flatland.envs.rail_env.RailEnv(
                width=map_size,
                height=map_size,
                number_of_agents=core_env.n_trains,
                rail_generator=core_env._rail_generator,
                malfunction_generator=core_env._malfunction_generator,
                line_generator=core_env._line_generator,
            )

            # -- Seed envs --

            maze_env.seed(1234)
            env.seed(1234)
            rail_env.reset(1234)
            _ = maze_env.reset()

            # -- Prepare action --

            maze_action = {'train_move': 2}
            maze_state = maze_env.get_maze_state()
            env_action = maze_env.action_conversion.space_to_maze(maze_action, maze_state)
            rail_env_action = {0: 0}

            start = time.time()
            for _ in range(n_iterations):
                maze_env.reset()

                for _ in range(n_env_steps):
                    obs, rewards, done, info = maze_env.step(maze_action)
            elapsed = time.time() - start

            print('MAZE ENV')
            print(f'- {n_iterations * n_env_steps} iterations done in {elapsed:.2f} seconds')
            print(f'- {n_iterations * n_env_steps / elapsed:.2f} iterations Per second')

            start = time.time()
            for _ in range(n_iterations):
                env.reset()

                for _ in range(n_env_steps):
                    obs, rewards, done, info = env.step(env_action)
            elapsed = time.time() - start

            print('\nCORE ENV')
            print(f'- {n_iterations * n_env_steps} iterations done in {elapsed:.2f} seconds')
            print(f'- {n_iterations * n_env_steps / elapsed:.2f} iterations Per second')

            start = time.time()
            for _ in range(n_iterations):
                rail_env.reset()

                for _ in range(n_env_steps):
                    obs, rewards, done, info = rail_env.step(rail_env_action)
            elapsed = time.time() - start

            print('\nRAIL ENV')
            print(f'- {n_iterations * n_env_steps} iterations done in {elapsed:.2f} seconds')
            print(f'- {n_iterations * n_env_steps / elapsed:.2f} iterations Per second\n')

        print(f'{"".join(["-"] * 30)}\n\n')


def main_serialization_benchmark(n_iterations: int, n_trains: list[int], map_sizes: list[int]) -> None:
    print('\n\nBasic Benchmarking...')
    print(f'Number of iterations: {n_iterations}')
    print(f'Number of trains: [{", ".join(map(str, n_trains))}]')
    print(f'Map sizes: [{", ".join(map(str, map_sizes))}]\n\n')

    for map_size in map_sizes:
        print(f'Mapsize {map_size}\n')

        for n_train in n_trains:
            print(f'Trains {n_train}\n')

            # -- MAZE ENV --

            core_env = init_core_env(map_size, map_size, n_train, 3, 1 / 100, speed_ratio_map={1: 1})
            maze_env = FlatlandEnvironment(
                core_env=core_env,
                action_conversion={'train_move': DirectionalAC()},
                observation_conversion={'train_move': PositionalObservationConversion(False)},
            )

            # -- Seed and reset envs --

            maze_env.seed(1234)
            maze_env.reset()

            # -- Start benchmarking --

            start = time.time()
            for _ in range(n_iterations):
                serialize_state = maze_env.serialize_state()
            elapsed = time.time() - start

            print('serialize_state')
            print(f'- {n_iterations} iterations done in {elapsed:.2f} seconds')
            print(f'- {n_iterations / elapsed:.2f} iterations Per second\n')

            start = time.time()
            for _ in range(n_iterations):
                maze_env.deserialize_state(serialize_state)
            elapsed = time.time() - start

            print('deserialize_state')
            print(f'- {n_iterations} iterations done in {elapsed:.2f} seconds')
            print(f'- {n_iterations / elapsed:.2f} iterations Per second\n')

        print(f'{"".join(["-"] * 30)}\n\n')


@profile
def main_serialization_benchmark_space(n_iterations: int, n_trains: list[int], map_sizes: list[int]) -> None:
    print('\n\nBasic Benchmarking...')
    print(f'Number of iterations: {n_iterations}')
    print(f'Number of trains: [{", ".join(map(str, n_trains))}]')
    print(f'Map sizes: [{", ".join(map(str, map_sizes))}]\n\n')

    for map_size in map_sizes:
        print(f'Mapsize {map_size}\n')

        for n_train in n_trains:
            print(f'Trains {n_train}\n')

            # -- MAZE ENV --

            core_env = init_core_env(
                map_size,
                map_size,
                n_train,
                3,
                1 / 100,
                speed_ratio_map={1: 1},
                include_maze_state_in_serialization=False,
            )
            maze_env = FlatlandEnvironment(
                core_env=core_env,
                action_conversion={'train_move': DirectionalAC()},
                observation_conversion={'train_move': PositionalObservationConversion(False)},
            )

            # -- Seed and reset envs --

            maze_env.seed(1234)
            maze_env.reset()

            # -- Start benchmarking --
            # -- Serialization WITHOUT state --

            for _ in range(n_iterations):
                serialized_state = maze_env.serialize_state()

            print('serialize_state done')

            for _ in range(n_iterations):
                maze_env.deserialize_state(serialized_state)

            print('deserialize_state done\n')

            # -- MAZE ENV --

            core_env_ = init_core_env(
                map_size,
                map_size,
                n_train,
                3,
                1 / 100,
                speed_ratio_map={1: 1},
                include_maze_state_in_serialization=True,
            )
            maze_env_ = FlatlandEnvironment(
                core_env=core_env_,
                action_conversion={'train_move': DirectionalAC()},
                observation_conversion={'train_move': PositionalObservationConversion(False)},
            )

            # -- Seed and reset envs --

            maze_env_.seed(1234)
            maze_env_.reset()

            # -- Serialization WITH state --

            for _ in range(n_iterations):
                serialized_state_ = maze_env_.serialize_state()

            print('serialize_state done')

            for _ in range(n_iterations):
                maze_env_.deserialize_state(serialized_state_)

            print('deserialize_state done\n')

        print(f'{"".join(["-"] * 30)}\n\n')


if __name__ == '__main__':
    # -- Benchmark different environments --

    N_ITERATIONS = 100
    N_ENV_STEPS = 20
    N_TRAINS = [1, 5, 10]
    MAP_SIZES = [35, 70]

    main_env_benchmark(N_ITERATIONS, N_ENV_STEPS, N_TRAINS, MAP_SIZES)

    # -- Benchmark serialization --

    N_ITERATIONS = 10
    N_TRAINS = [10]
    MAP_SIZES = [150]

    main_serialization_benchmark(N_ITERATIONS, N_TRAINS, MAP_SIZES)

    # -- Benchmark serialization space --

    N_ITERATIONS = 1
    N_TRAINS = [50]
    MAP_SIZES = [200]

    main_serialization_benchmark_space(N_ITERATIONS, N_TRAINS, MAP_SIZES)
