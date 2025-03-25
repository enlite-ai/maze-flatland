"""File holding scripts to quantify the probability of impossible episodes."""
from __future__ import annotations

import numpy as np
from maze_flatland.env.core_env import ImpossibleEpisodeException
from maze_flatland.test.env_instantation import create_core_env


def get_impossible_ratio(
    seeds: list[int],
    n_trains: int,
    map_size: tuple[int, int],
    n_cities: int,
    max_rails: int,
    max_parallel_rails_in_cities: int,
) -> tuple[float, list[int]]:
    """Estimates the  number of faulty episodes based on the given seeds.
    :param seeds: list of seeds to test.
    :param n_trains: number of trains to init an environment with
    :param map_size: tuple for the grid size to init the rail (width, height)
    :param n_cities: maximum number of cities to place on the grid.
    :param max_rails: maximum number of rails between cities
    :param max_parallel_rails_in_cities: maximum number of parallel tracks within each city.
    :return: ratio of faulty episodes in [0,1] and list of faulty seeds.
    """
    faulty_seeds = []
    env = create_core_env(
        n_trains,
        map_size[0],
        map_size[1],
        n_cities,
        0,
        {1.0: 1},
        False,
        max_rails,
        max_parallel_rails_in_cities,
    )
    for seed in seeds:
        env.seed(seed)
        try:
            env.reset()
        except ImpossibleEpisodeException:
            faulty_seeds.append(seed)

    return round(len(faulty_seeds) / len(seeds), 4), faulty_seeds


if __name__ == '__main__':
    base_seed = 1234
    np.random.seed(base_seed)
    n_runs = 100
    random_seeds = [np.random.randint(0, np.iinfo(np.int32).max) for _ in range(n_runs - 1)]
    random_seeds.append(1954)

    for train_counts in [1, 3, 5]:
        for grid_size in [(37, 37), (50, 50), (100, 100)]:
            for max_cities in [2, 5]:
                for max_rails_in_city in [2, 3]:
                    for max_rails_between_cities in [2, 3]:
                        fault_ratio, failing_seeds = get_impossible_ratio(
                            random_seeds,
                            train_counts,
                            grid_size,
                            max_cities,
                            max_rails_in_city,
                            max_rails_between_cities,
                        )
                        if fault_ratio > 0:
                            print('*' * 100)
                            print(
                                f'[ENV SETUP] n_trains: {train_counts}, map_size: {grid_size}, '
                                f'n_cities: {max_cities}, max_rail_pairs_in_city: {max_rails_in_city}, '
                                f'max_rails_between_cities: {max_rails_between_cities}\n'
                            )
                            print(f'{fault_ratio}% faulty episodes\n[SEEDS] {failing_seeds}')
