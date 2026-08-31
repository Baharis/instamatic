from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from instamatic._typing import AnyPath, float_nm, int_nm
from instamatic.grid import Intercepts
from instamatic.grid.grid import GRID_REGISTRY, PeriodicConvexPolygonGrid
from instamatic.grid.sweeping import star_sweep
from instamatic.gui.click_dispatcher import ClickListener, MouseButton


class GridFinder:
    """Base strategy for determining and updating grid geometry.
    Can be stored in a yaml file in the following format:

    grid_type: square
    geometry:
        x: 11.111
        y: 22.222
        t: 0.033  # degrees
        w: 44.444
        s: 6.666
    intercepts:
        0:
          - [0.001, 0.002]
          - [5.003, 5.004]
          # ...
        -1: # fresh, not assigned to a window yet
          - [99.005, 99.006]
    """

    GRID_REGISTRY_INV = {v: k for k, v in GRID_REGISTRY.items()}

    def __init__(
        self,
        grid: Optional[PeriodicConvexPolygonGrid] = None,
        intercepts: Optional[Intercepts] = None,
    ) -> None:
        self.grid = grid or GRID_REGISTRY['square'](0, 0, 0, 50_000, 50_000)
        self.intercepts: Intercepts = intercepts or {}
        self.path: Optional[AnyPath] = None  # if present, auto-save changes here

    @classmethod
    def from_yaml(cls, yaml_path: AnyPath) -> GridFinder:
        with open(Path(yaml_path), 'r') as f:
            data = yaml.safe_load(f)
        grid = GRID_REGISTRY[data['grid_type']](**data['geometry'])
        intercepts = data.get('intercepts', {})
        return cls(grid, {k: np.array(v, dtype=float) for k, v in intercepts})

    def to_yaml(self, yaml_path: AnyPath) -> None:
        grid_type_name = self.GRID_REGISTRY_INV[type(self.grid)]
        data = {
            'grid_type': grid_type_name,
            'geometry': self.grid.to_params(),
            'intercepts': {k: v.tolist() for k, v in self.intercepts.items()},
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)

    def add_intercept(self, window_idx: int, x: float_nm, y: float_nm) -> None:
        """Register a new intercept of given id, x, and y in the finder."""
        if window_idx in self.intercepts:
            self.intercepts[window_idx] = np.vstack([self.intercepts[window_idx], [x, y]])
        else:
            self.intercepts[window_idx] = np.array([[x, y]], dtype=float)
        if self.path is not None:
            self.to_yaml(self.path)

    def fit_intercepts(self, window_idx: int) -> None:
        """Fit all intercepts with given window id to a new window."""
        xy = self.intercepts[window_idx]
        if 0 in self.intercepts:
            new_center = (np.max(xy, axis=0) + np.min(xy, axis=0)) / 2
            new_window_idx = self.grid.nearest_index(*new_center)
        else:
            new_window_idx = 0
            self.grid = type(self.grid).guess({0: xy})
        new_intercepts = self.intercepts[window_idx]
        del self.intercepts[window_idx]
        self.intercepts[new_window_idx] = new_intercepts
        self.grid.refine(self.intercepts)
        if self.path is not None:
            self.to_yaml(self.path)

    def refine_by_manual_clicking(self, ctrl, cl: ClickListener) -> None:
        """Update grid & intercepts via clicks when stage is at window edge."""
        print('Please navigate the stage to as many points on one windows edge as possible')
        print('(at least the corners and midpoints). At each point, position the edge at')
        print('the center of the screen and LMB to add the point. RMB to finish.')
        while True:
            prev_grid, prev_intercepts = self.grid, self.intercepts
            with cl:
                while True:
                    c = cl.get_click()
                    if c.button == MouseButton.RIGHT:
                        break
                    self.add_intercept(-1, *ctrl.stage.xy)

            self.fit_intercepts(-1)
            print('Intercepts fit: LMB to accept, RMB to retry, MMB for new window')
            c = cl.get_click()
            if c.button == MouseButton.LEFT:
                break
            elif c.button == MouseButton.RIGHT:
                self.grid, self.intercepts = prev_grid, prev_intercepts
                if self.path is not None:
                    self.to_yaml(self.path)

    def refine_by_auto_sweeping(
        self,
        ctrl,
        window_idx: int = -1,
        x_lim: Optional[int_nm] = None,
        y_lim: Optional[int_nm] = None,
    ) -> None:
        """Let grid & intercepts refine by automatically looking for edges."""
        idx = window_idx
        if not self.intercepts:
            idx = 0
        else:
            d_lim = (sqrt(max(self.intercepts)) + 2) * (self.grid.w + self.grid.h)
            x_lim = x_lim or d_lim  # crude estimate of new window search area
            y_lim = y_lim or d_lim  # if no limits was given: (sqrt(idx)+2)(w+h)
            idc_in_limits = self.grid.windows_in_limits(x=x_lim, y=y_lim)
            if idx == -1:
                try:
                    idx = min([i for i in idc_in_limits if i not in self.intercepts])
                except ValueError:
                    raise IndexError('Could not locate next window within limits')
            else:
                if idx not in idc_in_limits:
                    raise IndexError(f'Requested window {idx} is not within limits')

        ctrl.stage.set(*[int(xy) for xy in self.grid.window(idx).center])
        ss_order = 3 if idx == 0 else 2 if len(self.intercepts.keys()) < 4 else 1
        new_intercepts = star_sweep(arms=3, order=ss_order, offset=17 * idx)
        for xy in new_intercepts:
            self.add_intercept(idx, *xy)
        self.fit_intercepts(idx)
