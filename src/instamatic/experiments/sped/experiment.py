from __future__ import annotations

from typing import Union

import numpy as np

from instamatic.calibrate import CalibMovieDelays
from instamatic.calibrate.calibrate_stage_translation import CalibStageTranslationX
from instamatic.experiments.experiment_base import ExperimentBase
from instamatic.experiments.fast_adt.experiment import FastADTMissingCalibError
from instamatic.grid.window import RectangularWindow


class Experiment(ExperimentBase):
    name = 'SPED'

    def __init__(self, ctrl, **kwargs):
        super().__init__()
        self.ctrl = ctrl

        self.exposure = 0.1
        self.speed: Union[float, int] = 1.0
        self.xy_resolution = 2

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

    def determine_translation_speed(self) -> None:
        detector_dead_time = self.get_dead_time(self.exposure)
        time_for_one_frame = self.exposure + detector_dead_time
        trans_calib = self.get_stage_translation()
        mot_plan = trans_calib.plan_motion(time_for_one_frame / self.xy_resolution)
        self.exposure = abs(mot_plan.pace * self.xy_resolution) - detector_dead_time
        self.speed = mot_plan.speed

    def start_collection(self, **params) -> None:
        # precalculate sliding speeds
        self.determine_translation_speed()

        # plan the scanning of current grid window
        win = RectangularWindow.from_sweeping(order=3)
        y = np.min(win.corners[:, 1]) + (0.5 * self.xy_resolution)
        scans: dict[int, tuple[float, float]] = {}
        for i, x in enumerate(win.x_intersections(y)):
            if x is None:
                break
            scans[y] = (x[0], x[1]) if i % 2 else (x[1], x[0])
            y += self.xy_resolution

        # for each scan, collect a movie
        for y, (x0, x1) in scans.items():
            self.ctrl.stage.set(x=x0, y=y)
            x_n = int(np.ceil(abs(x1 - x0) / self.xy_resolution))
            movie = self.ctrl.get_movie(n_frames=x_n, exposure=self.exposure)
            self.ctrl.stage.set_with_speed(x=x1, speed=self.speed)
            for x_i, (image, meta) in enumerate(movie):
                ...  # send image to multiprocessor analyzer

        return

    def finalize(self) -> None: ...
