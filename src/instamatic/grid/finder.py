from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np

from instamatic._typing import float_nm, int_nm
from instamatic.grid.geometry import GRID_REGISTRY
from instamatic.grid.logger import GridLogger
from instamatic.grid.sweeping import star_sweep
from instamatic.gui.click_dispatcher import ClickListener, MouseButton

Intercepts = dict[int, np.ndarray]


class GridFinder:
    """Base strategy for determining and updating grid geometry."""

    def __init__(self, logger: GridLogger):
        self.logger: GridLogger = logger
        self.grid = GRID_REGISTRY['square'](0, 0, 0, 50_000, 50_000)
        self.intercepts: Intercepts = {}

    def add_intercept(self, window_idx: int, x: float_nm, y: float_nm) -> None:
        """Register a new intercept of given id, x, and y in the finder."""
        if window_idx in self.intercepts:
            self.intercepts[window_idx] = np.vstack([self.intercepts[window_idx], [x, y]])
        else:
            self.intercepts[window_idx] = np.array([x, y], dtype=float)
        self.logger.write(self.grid, self.intercepts)

    def fit_intercepts(self, window_idx: int) -> None:
        """Fit all intercepts with given window id to a new window."""
        xy = self.intercepts[window_idx]
        if 0 in self.intercepts:
            new_center = (np.max(xy, axis=0) - np.min(xy, axis=0)) / 2
            new_window_idx = self.grid.nearest_index(*new_center)
        else:
            new_window_idx = 0
            self.grid.guess(xy)
        new_intercepts = self.intercepts[window_idx]
        del self.intercepts[window_idx]
        self.intercepts[new_window_idx] = new_intercepts
        self.grid.refine(self.intercepts)
        self.logger.write(self.grid, self.intercepts)

    def refine_by_manual_clicking(self, ctrl, cl: ClickListener) -> None:
        """Update grid & intercepts via clicks when stage is at window edge."""
        print('Navigate to points on one window edge. LMB to add, RMB to finish.')
        while True:
            prev_grid, prev_intercepts = self.grid, self.intercepts
            with cl:
                while True:
                    c = cl.get_click()
                    if c.button == MouseButton.RIGHT:
                        break
                    self.add_intercept(-1, *ctrl.stage.xy)

            self.fit_intercepts(-1)
            print('LMB to accept, RMB to retry, MMB for new window')
            c = cl.get_click()
            if c.button == MouseButton.LEFT:
                break
            elif c.button == MouseButton.RIGHT:
                self.grid, self.intercepts = prev_grid, prev_intercepts
                self.logger.write(self.grid, self.intercepts)

    def refine_by_auto_sweeping(
        self,
        ctrl,
        x_lim: Optional[int_nm] = None,
        y_lim: Optional[int_nm] = None,
    ) -> None:
        """Let grid & intercepts refine by automatically looking for edges."""
        if not self.intercepts:
            idx = 0
        else:
            d_lim = (sqrt(max(self.intercepts)) + 2) * (self.grid.w + self.grid.h)
            x_lim = x_lim or d_lim  # crude estimate of new window search area
            y_lim = y_lim or d_lim  # if no limits was given: (sqrt(idx)+2)(w+h)
            idc_in_limits = self.grid.windows_in_limits(x=x_lim, y=y_lim)
            try:
                idx = min([i for i in idc_in_limits if i not in self.intercepts])
            except ValueError:
                raise IndexError('Could not locate next window within limits')

        ctrl.stage.set(*[int(xy) for xy in self.grid.window(idx).center])
        ss_order = 3 if idx == 0 else 2 if len(self.intercepts.keys()) < 4 else 1
        new_intercepts = star_sweep(arms=3, order=ss_order, offset=17 * idx)
        for xy in new_intercepts:
            self.add_intercept(idx, *xy)
        self.fit_intercepts(idx)
