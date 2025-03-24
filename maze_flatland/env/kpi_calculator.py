"""KPI calculator for the Flatland environment."""


from __future__ import annotations

from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.kpi_calculator import KpiCalculator


class FlatlandKPICalculator(KpiCalculator):
    """
    KPI calculator for Flatland environment. As of yet no KPIs are available.
    """

    def calculate_kpis(self, episode_event_log: EpisodeEventLog, last_maze_state: MazeStateType) -> dict[str, float]:
        """
        Implementation of :py:meth:`~maze.core.log_events.kpi_calculator.KpiCalculator.calculate_kpis`.
        """

        raise NotImplementedError
