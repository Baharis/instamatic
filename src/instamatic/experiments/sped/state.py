from __future__ import annotations

import numpy as np
import pandas as pd

from instamatic.grid.window import ConvexPolygonGridWindow


class SPEDState:
    """Stores the current state of the SPED experiment in history dataframe."""

    def __init__(self) -> None:
        self.grids: list[int] = []
        self.windows: dict[tuple[int, int], ConvexPolygonGridWindow] = {}
        self.scans: pd.DataFrame = pd.DataFrame()
        self.steps: pd.DataFrame = pd.DataFrame()
        self.init_dataframes()

    def init_dataframes(self) -> None:
        """Create a new empty history with required index and columns."""
        self.scans = pd.DataFrame(
            {
                'grid': pd.Series(dtype=np.uint8),
                'window': pd.Series(dtype=np.uint16),
                'scan': pd.Series(dtype=np.uint16),
                'x0': pd.Series(dtype=np.int32),
                'y0': pd.Series(dtype=np.int32),
                'direction': pd.Series(dtype=np.str_),
                'span': pd.Series(dtype=np.uint32),
            }
        )
        self.scans.set_index(['grid', 'window', 'scan'], inplace=True)
        self.steps = pd.DataFrame(
            {
                'grid': pd.Series(dtype=np.uint8),
                'window': pd.Series(dtype=np.uint16),
                'scan': pd.Series(dtype=np.uint16),
                'step': pd.Series(dtype=np.uint16),
                'success': pd.Series(dtype=pd.BooleanDtype),
                'n_peaks': pd.Series(dtype=np.uint16),
            }
        )
        self.steps.set_index(['grid', 'window', 'scan', 'step'], inplace=True)

    def add_scan(self, grid: int, window: int, scan: int, n_frames: int) -> None:
        """Pre-allocate scan space in the history dataframe for performance."""
        new = pd.DataFrame(
            {
                'grid': np.full(n_frames, grid, dtype=np.uint8),
                'window': np.full(n_frames, window, dtype=np.uint16),
                'scan': np.full(n_frames, scan, dtype=np.uint16),
                'step': np.arange(n_frames, dtype=np.uint16),
                'success': pd.array([pd.NA] * n_frames, dtype='boolean'),
                'n_peaks': np.zeros(n_frames, dtype=np.uint16),
            }
        )
        new.set_index(['grid', 'window', 'scan', 'step'], inplace=True)
        self.steps = pd.concat([self.steps, new], copy=False)

    def add_step(self, idx, *, success: bool, n_peaks: int) -> None:
        """Add a single result line to the history dataframe."""
        self.steps.at[idx, 'success'] = bool(success)
        self.steps.at[idx, 'n_peaks'] = int(n_peaks)
