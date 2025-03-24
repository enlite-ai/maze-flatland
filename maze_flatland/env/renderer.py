"""
Renderer for Flatland environment.
"""
from __future__ import annotations

from typing import Optional

import flatland.envs.rail_env
import flatland.utils.rendertools
import matplotlib.pyplot as plt
import numpy as np
from maze.core.annotations import override
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer import Renderer
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import FlatlandMazeState


class FlatlandRendererBase(Renderer):
    """Base class for rendering the flatland environment.

    :param img_width: Width of the rendered image.
    :param agent_render_variant: Variant to render the agent on map.
    :param highlight_current_train: Whether to highlight current train or not.
    :param render_out_of_map_trains: Whether to render trains that have yet to depart.
    :param show_grid: Whether to show the grid. Default: False.
    """

    def __init__(
        self,
        img_width: int,
        agent_render_variant: flatland.utils.rendertools.AgentRenderVariant,
        highlight_current_train: bool,
        render_out_of_map_trains: bool,
        show_grid: bool = False,
    ):
        self._screen_width = img_width
        self._screen_height = img_width
        self._gl = 'PILSVG'
        self._flatland_renderer: Optional[flatland.utils.rendertools.RenderLocal] = None
        self.agent_render_variant = agent_render_variant
        self._show_selected_agent = highlight_current_train
        self._show_inactive_agents = render_out_of_map_trains
        self._show_grid = show_grid

    @property
    def is_initialized(self) -> bool:
        """
        Checks whether renderer has been initialized.

        :return: Whether renderer has been initialized.
        """
        return self._flatland_renderer is not None

    def reset(self) -> None:
        """Resets the renderer."""
        self._flatland_renderer = None

    def _init_flatland_render_tool(self, rail_env: flatland.envs.rail_env.RailEnv) -> None:
        """
        Initializes Flatland's RenderTool.

        :param rail_env: Flatland's RailEnv.
        :return: Initialized Flatland RenderTool.
        """
        self._screen_height = self._screen_width * (rail_env.height / rail_env.width)
        self._flatland_renderer = flatland.utils.rendertools.RenderLocal(
            rail_env,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            agent_render_variant=self.agent_render_variant,
            gl=self._gl,
        )

    def close(self) -> None:
        """
        Closes renderer and all Matplotlib plots.
        """

        plt.close('all')
        if self._flatland_renderer:
            self._flatland_renderer.close_window()

    @override(Renderer)
    def render(
        self,
        maze_state: FlatlandMazeState,
        maze_action: None | FlatlandMazeAction | list[FlatlandMazeAction],
        events: StepEventLog,
        rail_env: flatland.envs.rail_env.RailEnv,
        close_prior_windows: bool,
        save_in_dir: Optional[str] = None,
        override_fsize: None | tuple[int, int] = None,
    ) -> None:
        """
        Implementation of :py:meth:`~maze.core.rendering.renderer.Renderer.render`.

        :param maze_state: Maze state after application of action.
        :param maze_action: The optional action to render. Or a list of actions for each train.
        :param events: StepEventLog instance.
        :param rail_env: Flatland's RailEnv.
        :param close_prior_windows: Whether to close prior windows before showing new one.
        :param save_in_dir: Optional dir path to save the image in. If None show the image.
        :param override_fsize: If defined, it overrides the standard image size.
        """
        if maze_state.terminate_episode:
            return
        _ = maze_action, events
        _ = self._render_base_grid(
            maze_state,
            rail_env,
            self._show_inactive_agents,
            self._show_selected_agent,
            True,
            close_prior_windows,
            override_fsize,
        )

        if save_in_dir is None:
            plt.show()
        else:
            plt.savefig(f'{save_in_dir}/render_{maze_state.env_time}-{maze_state.current_train_id}.png')
            plt.close()

    def _render_base_grid(
        self,
        maze_state: FlatlandMazeState,
        rail_env: flatland.envs.rail_env.RailEnv,
        show_inactive_agent: bool,
        select_agent: bool,
        show_agents: bool,
        close_prior_windows: bool,
        override_fsize: None | tuple[int, int] = None,
    ) -> np.ndarray:
        """Helper method to do the rendering from the flatland environment.

        :param maze_state: Maze state after application of action.
        :param rail_env: Flatland's RailEnv.
        :param show_inactive_agent: Whether to show out of map trains.
        :param select_agent: If true then the current agent is highlighted.
        :param show_agents: If true then the agent are rendered.
        :param close_prior_windows: Whether to close prior windows before showing new one.
        :param override_fsize: If defined, it overrides the standard image size.
        :return: The rendered image.
        '"""
        # Lazy initialization of Flatland's integrated renderer.
        if not self.is_initialized or rail_env != self._flatland_renderer.env:
            self._init_flatland_render_tool(rail_env)
        selected_agent = maze_state.current_train_id if select_agent else None
        # self._flatland_renderer.close_window()
        img = self._flatland_renderer.render_env(
            return_image=True,
            show_inactive_agents=show_inactive_agent,
            show_observations=False,
            show_predictions=False,
            show_rowcols=True,
            show_agents=show_agents,
            selected_agent=selected_agent,
        )
        if close_prior_windows:
            plt.cla()
            plt.close('all')

        # Plot grid
        dpi = 100
        fsize = (self._screen_width / dpi, self._screen_height / dpi) if override_fsize is None else override_fsize
        plt.figure(figsize=fsize, dpi=dpi)
        plt.imshow(img, aspect='auto', interpolation='nearest')

        # Show timestamp
        plt.text(
            20,
            40,
            f'{maze_state.env_time}-{maze_state.current_train_id} / {maze_state.max_episode_steps}',
            fontsize=30,
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5},
        )

        if self._show_grid:
            for y in np.arange(0, img.shape[0], img.shape[0] // maze_state.map_size[1]):
                plt.axhline(y=y, color='red', alpha=0.75)
            for x in np.arange(0, img.shape[1], img.shape[0] // maze_state.map_size[0]):
                plt.axvline(x=x, color='red', alpha=0.75)

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        return img
