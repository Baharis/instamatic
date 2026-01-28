from __future__ import annotations

import logging
from itertools import cycle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from instamatic._typing import AnyPath, int_nm
from instamatic.calibrate import CalibMovieDelays
from instamatic.calibrate.calibrate_stage_translation import CalibStageTranslationX
from instamatic.experiments.experiment_base import ExperimentBase
from instamatic.experiments.fast_adt.experiment import FastADTMissingCalibError
from instamatic.experiments.scan_ed.dispatch import DiffHuntDispatcher
from instamatic.experiments.scan_ed.journal import Journal
from instamatic.experiments.scan_ed.progress import ProgressTable
from instamatic.experiments.scan_ed.state import State
from instamatic.grid.grid import ConvexPolygonGrid
from instamatic.grid.window import ConvexPolygonWindow, RectangularWindow


class Experiment(ExperimentBase):
    name = 'SPED'

    def __init__(
        self,
        ctrl,
        path: AnyPath,
        log: logging.Logger,
        flatfield: Optional[np.ndarray] = None,
        progress: Optional[ProgressTable] = None,
        load: bool = False,
    ):
        super().__init__()
        self.ctrl = ctrl
        self.path: Path = Path(path)
        self.log: logging.Logger = log
        self.flatfield: Optional[np.ndarray] = flatfield
        self.state = self.get_state(load=load, progress=progress)

        self.dispatcher = DiffHuntDispatcher(shape=(514, 514), dtype=np.uint16)
        self.dispatcher.initialize_workers()
        # TODO init the dispatcher correctly once actual camera size is known

    def get_dead_time(
        self,
        exposure: float = 0.0,
        header_keys_variable: tuple = (),
        header_keys_common: tuple = (),
    ) -> float:
        """Get time between get_movie frames from any source available or 0."""
        try:
            return self.ctrl.cam.dead_time
        except AttributeError:
            pass
        print('`cam.dead_time` not found. Looking for calibrated estimate...')
        try:
            c = CalibMovieDelays.from_file(exposure, header_keys_variable, header_keys_common)
        except RuntimeWarning:
            return 0.0
        else:
            return c.dead_time

    def get_stage_translation(self) -> CalibStageTranslationX:
        """Get rotation calibration if present; otherwise warn & terminate."""
        try:
            return CalibStageTranslationX.from_file()
        except OSError:
            print(m1 := 'This script requires stage rotation to be calibrated.')
            print(m2 := 'Please run `instamatic.calibrate_stage_rotation` first.')
            raise FastADTMissingCalibError(m1 + ' ' + m2)

    def get_state(self, load: bool, progress: Optional[ProgressTable] = None) -> State:
        """Initialize a state, fill it from journal; raise at load issues."""
        journal_path = self.path / 'journal.jsonl'
        journal = Journal(path=journal_path)
        state = State(journal=journal, progress=progress)
        if load:
            if not journal_path.exists() or not journal_path.is_file():
                raise FileNotFoundError(f'No journal file found at {journal_path=}')
            state.load_from_journal()
        return state

    def get_grid(self, params: dict[str, Any]) -> ConvexPolygonGrid:
        """Reconstruct the grid from current params and state."""
        from instamatic.grid.grid import HexagonalGrid, RectangularGrid

        if params.get('grid_geometry', '').lower().startswith('hex'):
            grid = HexagonalGrid()
        else:
            grid = RectangularGrid()
        if self.state.windows:
            for wid, w in self.state.windows.items():
                assert isinstance(w, grid.window_type)
                grid.windows[wid] = w
        return grid

    def determine_exposure_and_speed(
        self,
        exposure: float,
        step_size: int_nm,
    ) -> tuple[float, float]:
        detector_dead_time = self.get_dead_time(exposure)
        time_for_one_frame = exposure + detector_dead_time
        trans_calib = self.get_stage_translation()
        mot_plan = trans_calib.plan_motion(1e9 * time_for_one_frame / step_size)
        exposure = abs(mot_plan.pace * step_size) - detector_dead_time
        speed = mot_plan.speed
        return exposure, speed

    def start_collection(self, **params) -> None:
        # precalculate sliding speeds
        exposure, speed = self.determine_exposure_and_speed(
            params['exposure'], params['step_size']
        )
        grid = self.get_grid(params=params)
        stop_event = params['stop_event']

        while not stop_event.is_set():
            try:
                window_id, window = self.locate_next_window(grid=grid, params=params)
            except IndexError:
                break
            grid.windows['window_id'] = window
            self.state.add_window(idx=window_id, window=window)

            if params['scan_geometry'].lower().startswith('x'):
                fast_axis, scan_factory = 'x', window.x_intersections
                fast_step, slow_step = params['scan_x_step'], params['scan_y_step']
                slow_axis_idx = 1
            else:  # params['scan_geometry'].lower().startswith('y'):
                fast_axis, scan_factory = 'y', window.y_intersections
                fast_step, slow_step = params['scan_y_step'], params['scan_x_step']
                slow_axis_idx = 0

            if params['scan_geometry'].lower().endswith('raster'):
                scan_signs = cycle([1, -1])
            else:  # params['scan_geometry'].lower().endswith('raster'):
                scan_signs = [
                    1,
                ]

            slow_min = np.min(window.corners[:, slow_axis_idx])
            slow_max = np.max(window.corners[:, slow_axis_idx])
            for scan_id, slow in enumerate(
                np.arange(slow_min + slow_step, slow_max, slow_step)
            ):
                fast_min, fast_max = scan_factory(float(slow))
                self.state.add_scan(
                    window=window_id,
                    scan_id=scan_id,
                    x0=fast_min if fast_axis == 'x' else slow_min,
                    y0=fast_min if fast_axis == 'y' else slow_min,
                    direction=('+' if next(scan_signs) >= 0 else '-') + fast_axis,
                    span=abs(fast_max - fast_min),
                    step=fast_step,
                )

            for scan_id in self.state.scans.loc[window_id].index:
                idx = pd.IndexSlice[window_id, scan_id, :]
                if self.state.steps.loc[idx, 'success'].notna().any():
                    continue  # this scan has been already done
                x0 = self.state.scans.at[(window_id, scan_id), 'x0']
                y0 = self.state.scans.at[(window_id, scan_id), 'y0']
                self.ctrl.stage.set(x0=x0, y0=y0)

                movie = self.ctrl.get_movie(
                    n_frames=len(idx), exposure=exposure, header_keys=None
                )
                self.dispatcher.switch_buffer(len(idx), name=f'w:{window_id}/s:{scan_id}')
                span = self.state.scans.at[(window_id, scan_id), 'span']
                direction = self.state.scans.at[(window_id, scan_id), 'direction']
                sign = +1 if direction.startswith('+') else -1
                fast1 = (x0 if direction.endswith('X') else y0) + sign * span
                setter_kwargs = {fast_axis: fast1, 'speed': speed}

                self.ctrl.stage.set_with_speed(**setter_kwargs)
                for frame, header in movie:
                    self.dispatcher.process(frame, header)
                    # TODO: receive all dispatch feedback
                    # TODO somehow wait until entire buffer is filled
                    self.dispatcher.write_buffer(self.path)
                    # TODO: write from history to state
                    # TODO: new writing path for every scan

            # TODO: add missing logic, repeated scans
            # TODO: state: replace windows list with grid to avoid duplication
            # TODO: simplify scans logic because right now it is difficult

        return

    def locate_next_window(
        self,
        grid: ConvexPolygonGrid,
        params: dict,
    ) -> tuple[int, ConvexPolygonWindow]:
        """Find a next window on the grid, or raise if none can be found."""
        last_window_id = max(grid.windows)
        for window_id in range(last_window_id + 1, 2 * last_window_id + 10):
            predicted = grid.predict_window(window_id)
            x_lim = tx if (tx := params['target_x']) is not None else float('inf')
            y_lim = ty if (ty := params['target_x']) is not None else float('inf')
            x_fits = np.all(np.abs(predicted.corners[:, 0]) < x_lim)
            y_fits = np.all(np.abs(predicted.corners[:, 0]) < y_lim)
            if not (x_fits and y_fits):
                continue
            self.ctrl.stage.set(*[int(xy) for xy in predicted.center])
            return window_id, grid.window_type.from_sweeping()
        raise IndexError('Could not locate next window within limits')

    def finalize(self) -> None:
        ...
        # TODO
