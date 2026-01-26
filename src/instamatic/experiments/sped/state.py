from __future__ import annotations

import numpy as np
import pandas as pd

from instamatic.grid.window import ConvexPolygonWindow


class SPEDState:
    """Stores the current state of the SPED experiment in history dataframe."""

    def __init__(self) -> None:
        self.windows: dict[int, ConvexPolygonWindow] = {}
        self.scans: pd.DataFrame = pd.DataFrame()
        self.steps: pd.DataFrame = pd.DataFrame()
        self.init_dataframes()

    def init_dataframes(self) -> None:
        """Create a new empty history with required index and columns."""
        self.scans = pd.DataFrame(
            {
                'window': pd.Series(dtype=np.uint16),
                'scan': pd.Series(dtype=np.uint16),
                'x0': pd.Series(dtype=np.int32),
                'y0': pd.Series(dtype=np.int32),
                'direction': pd.Series(dtype=np.str_),
                'span': pd.Series(dtype=np.uint32),
            }
        )
        self.scans.set_index(['window', 'scan'], inplace=True)
        self.steps = pd.DataFrame(
            {
                'window': pd.Series(dtype=np.uint16),
                'scan': pd.Series(dtype=np.uint16),
                'step': pd.Series(dtype=np.uint16),
                'success': pd.Series(dtype=pd.BooleanDtype),
                'n_peaks': pd.Series(dtype=np.uint16),
            }
        )
        self.steps.set_index(['window', 'scan', 'step'], inplace=True)

    def add_window(self, idx: int, window: ConvexPolygonWindow) -> None:
        self.windows[idx] = window

    def add_scan(
        self,
        window: int,
        scan: int,
        x0: int,
        y0: int,
        direction: str,
        span: int,
        n_frames: int,
    ) -> None:
        """Append to scans and pre-allocate space in the steps dataframe."""
        new_scan = {'x0': x0, 'y0': y0, 'direction': direction, 'span': span}
        self.scans.loc[window, scan] = new_scan

        new_steps = pd.DataFrame(
            {
                'window': np.full(n_frames, window, dtype=np.uint16),
                'scan': np.full(n_frames, scan, dtype=np.uint16),
                'step': np.arange(n_frames, dtype=np.uint16),
                'success': pd.array([pd.NA] * n_frames, dtype='boolean'),
                'n_peaks': np.zeros(n_frames, dtype=np.uint16),
            }
        )
        new_steps.set_index(['window', 'scan', 'step'], inplace=True)
        self.steps = pd.concat([self.steps, new_steps], copy=False)

    def fill_scan(
        self,
        window: int,
        scan: int,
        success: np.ndarray,
        n_peaks: np.ndarray,
    ) -> None:
        """Fill a previously-added scan with success/n_peaks in one update."""
        idx = pd.IndexSlice[window, scan, :]

        n_rows = self.steps.loc[idx].shape[0]
        if len(success) != n_rows or len(n_peaks) != n_rows:
            raise ValueError(f'Expected {n_rows} steps, got {len(success)=}, {len(n_peaks)=}')

        self.steps.loc[idx, 'success'] = pd.array(success, dtype='boolean')
        self.steps.loc[idx, 'n_peaks'] = np.asarray(n_peaks, dtype=np.uint16)
