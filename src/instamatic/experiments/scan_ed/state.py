from __future__ import annotations

import ast
import importlib
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from instamatic.experiments.scan_ed.journal import Journal, edits_journal
from instamatic.experiments.scan_ed.progress import ProgressTable, edits_progress
from instamatic.grid.window import ConvexPolygonWindow

WindowFactory: Callable[[float, float, float, ...], type[ConvexPolygonWindow]]


class State:
    """Stores the current state of the SPED experiment in history dataframe."""

    def __init__(self, journal: Journal, progress: Optional[ProgressTable] = None) -> None:
        self.journal: Journal = journal
        self.progress: ProgressTable = progress

        self.windows: dict[int, ConvexPolygonWindow] = {}
        self.scans: pd.DataFrame = pd.DataFrame()
        self.steps: pd.DataFrame = pd.DataFrame()
        self._init_dataframes()

    def _init_dataframes(self) -> None:
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

    def load_from_journal(self) -> None:
        with self.journal.writing_off():
            for event in self.journal.events():
                method = getattr(self, event['method'])
                kwargs = event.get('kwargs', {})
                method(**kwargs)

    @edits_journal
    @edits_progress
    def add_window(self, idx: int, window: Union[ConvexPolygonWindow, str]) -> None:
        """For journaling purposes, can be added via instance or __repr__."""
        if isinstance(window, str):
            body = ast.parse(window, mode='eval').body
            assert isinstance(body, ast.Call), f'Failed to eval "{window}"'
            assert isinstance(body.func, ast.Name), f'Failed to eval "{window}"'
            window_class_name = body.func.id
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in body.keywords}
            window_module = importlib.import_module('instamatic.grid.window')
            window_class = getattr(window_module, window_class_name)
            window = window_class(**kwargs)
        self.windows[idx] = window

    @edits_journal
    @edits_progress
    def add_scan(
        self,
        window: int,
        scan: int,
        x0: int,
        y0: int,
        direction: str,
        span: int,
        step: int,
    ) -> None:
        """Append to scans and pre-allocate space in the steps dataframe."""
        new_scan = {'x0': x0, 'y0': y0, 'direction': direction, 'span': span, 'step': step}
        self.scans.loc[window, scan] = new_scan

        n_steps = -(-span // step)
        new_steps = pd.DataFrame(
            {
                'window': np.full(n_steps, window, dtype=np.uint16),
                'scan': np.full(n_steps, scan, dtype=np.uint16),
                'step': np.arange(n_steps, dtype=np.uint16),
                'success': pd.array([pd.NA] * n_steps, dtype='boolean'),
                'n_peaks': np.zeros(n_steps, dtype=np.uint16),
            }
        )
        new_steps.set_index(['window', 'scan', 'step'], inplace=True)
        self.steps = pd.concat([self.steps, new_steps], copy=False)

    @edits_journal
    @edits_progress
    def fill_scan(
        self,
        window: int,
        scan: int,
        success: Union[np.ndarray, Sequence[Union[bool, None]]],
        n_peaks: Union[np.ndarray, Sequence[int]],
    ) -> None:
        """Fill a previously-added scan with success/n_peaks in one update."""
        idx = pd.IndexSlice[window, scan, :]

        n_rows = self.steps.loc[idx].shape[0]
        if len(success) != n_rows or len(n_peaks) != n_rows:
            raise ValueError(f'Expected {n_rows} steps, got {len(success)=}, {len(n_peaks)=}')

        self.steps.loc[idx, 'success'] = pd.array(success, dtype='boolean')
        self.steps.loc[idx, 'n_peaks'] = np.asarray(n_peaks, dtype=np.uint16)
