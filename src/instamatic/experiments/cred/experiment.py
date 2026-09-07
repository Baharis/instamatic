from __future__ import annotations

import datetime
import json
import os
import re
import socket
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

import numpy as np

import instamatic
from instamatic import config
from instamatic.experiments.experiment_base import ExperimentBase
from instamatic.formats import write_tiff

# -----------------------------------------------------------------------------
# Conversion backend
# -----------------------------------------------------------------------------
# This experiment is intended for a SerialEM -> DigitalMicrograph (DM) camera
# pipeline. It prefers the user-patched ImgConversionDM implementation.
#
# To use the patched converter:
#   1) Copy `ImgConversionDM_patched_v2.py` into:
#        <python env>/Lib/site-packages/instamatic/processing/
#   2) Keep the filename as-is (ImgConversionDM_patched_v2.py)
#   3) Restart Instamatic
try:
    from instamatic.processing.ImgConversionDM_patched_v2 import (
        ImgConversionDM as ImgConversion,
    )
except Exception:
    # Fallback to the installed DM converter
    from instamatic.processing.ImgConversionDM import ImgConversionDM as ImgConversion

# degrees to rotate before activating data collection procedure
ACTIVATION_THRESHOLD = 0.2

# When auto-stopping in footfree mode, consider target reached within this tolerance (deg)
END_ANGLE_TOLERANCE = 0.05

# Maximum time to wait for the stage to begin moving after issuing the rotation command
START_MOVE_TIMEOUT = 5.0  # seconds

# Consider the stage stalled (i.e. stopped) if the angle does not change by ANGLE_STALL_EPS
# for STAGE_STALL_TIMEOUT seconds. Used only for 'footfree' mode auto-stop.
STAGE_STALL_TIMEOUT = 0.5  # seconds
ANGLE_STALL_EPS = 0.01  # degrees

use_vm = getattr(getattr(config, 'settings', object()), 'use_VM_server_exe', False)


# -----------------------------------------------------------------------------
# Stage rotation speed calibration (footfree speed 0..1 -> deg/s)
# -----------------------------------------------------------------------------
# The calibration tool produces a fit like:
#   time = (pace * span + windup) / speed + delay
# where:
#   span  = rotation range in degrees
#   speed = footfree speed setting (0..1)
#   time  = total time in seconds
#
# We invert this model to choose a footfree speed setting that matches a
# requested rotation speed in deg/s for a given span.
#
# Calibration parameters are loaded from YAML files like `calib_stage_rotation.yaml`
# or `calib_stage_rotation_fit.yaml`. The user's calibration example contains:
#   pace, windup, delay.
#
# If no calibration file is found, the code will fall back to the explicit
# `footfree_speed` argument (0..1).
STAGE_ROT_CALIB_FILENAMES = (
    'calib_stage_rotation.yaml',
    'calib_stage_rotation_fit.yaml',
    'stage_rotation.yaml',
    'stage_rotation_fit.yaml',
)


@dataclass
class StageRotationCalib:
    pace: float
    windup: float
    delay: float
    lower_lim: float = 0.0
    upper_lim: float = 1.0

    @classmethod
    def from_yaml_dict(cls, d: dict) -> 'StageRotationCalib':
        # Two supported schemas:
        # - {pace, windup, delay, speed_options:{lower_lim,upper_lim}}
        # - {params:{pace,windup,delay}, ...}
        if 'params' in d:
            p = d.get('params', {})
            pace = float(p['pace'])
            windup = float(p['windup'])
            delay = float(p['delay'])
            lower_lim = float(d.get('speed_options', {}).get('lower_lim', 0.0))
            upper_lim = float(d.get('speed_options', {}).get('upper_lim', 1.0))
        else:
            pace = float(d['pace'])
            windup = float(d['windup'])
            delay = float(d['delay'])
            lower_lim = float(d.get('speed_options', {}).get('lower_lim', 0.0))
            upper_lim = float(d.get('speed_options', {}).get('upper_lim', 1.0))
        return cls(
            pace=pace, windup=windup, delay=delay, lower_lim=lower_lim, upper_lim=upper_lim
        )

    def predict_time_s(self, span_deg: float, speed_setting: float) -> float:
        span = abs(float(span_deg))
        speed = float(speed_setting)
        # Many FEI/Tecnai controllers become unreliable for extremely small nonzero speed values.
        # Enforce a practical minimum of 0.01 (unless the calibration lower limit is higher).
        min_speed = max(float(self.lower_lim), 0.01)
        max_speed = float(self.upper_lim)
        speed = max(min_speed, min(max_speed, speed))
        return (self.pace * span + self.windup) / max(speed, 1e-6) + self.delay

    def estimate_deg_per_s(self, span_deg: float, speed_setting: float) -> float:
        span = abs(float(span_deg))
        t = self.predict_time_s(span_deg=span, speed_setting=speed_setting)
        # Effective rotation time excludes the fixed delay term
        eff = max(t - self.delay, 1e-6)
        return span / eff

    def speed_for_deg_per_s(self, span_deg: float, target_deg_per_s: float) -> float:
        span = abs(float(span_deg))
        v = float(target_deg_per_s)
        if v <= 0:
            raise ValueError(f'Requested rotation speed must be > 0 deg/s (got {v}).')
        # t_target = span / v
        t_target = span / v
        denom = t_target - self.delay
        if denom <= 0:
            # Requested speed is unrealistically high for this model span.
            raise ValueError(
                f'Requested rotation speed {v} deg/s is too high for span={span} deg '
                f'(span/v={t_target:.3f}s <= delay={self.delay:.3f}s).'
            )
        speed = (self.pace * span + self.windup) / denom
        # Clamp to allowed bounds (with a practical minimum)
        min_speed = max(float(self.lower_lim), 0.01)
        max_speed = float(self.upper_lim)
        speed = max(min_speed, min(max_speed, speed))
        return speed


def _candidate_calibration_dirs() -> list[Path]:
    """Return a list of directories to search for the stage-rotation
    calibration YAML.

    The `instamatic.calibrate_stage_rotation` tool writes `calib_stage_rotation.yaml` by
    default into `%AppData%/calib` (Windows) unless an explicit `-o/--outdir` is given.

    In addition, many users keep this file alongside the Instamatic config directory
    (e.g. `%AppData%/instamatic/config`) or next to the experiment module.

    This function attempts to cover these common locations robustly across installs.
    """

    dirs: list[Path] = []

    # 1) Instamatic config module commonly exposes these paths
    for attr in (
        'base_drc',
        'config_drc',
        'calibration_drc',
        'microscope_drc',
        'camera_drc',
        'logs_drc',
        'scripts_drc',
    ):
        v = getattr(config, attr, None)
        if v:
            try:
                dirs.append(Path(v))
            except Exception:
                pass

    # 2) config.locations is a dict of important paths (newer Instamatic)
    locs = getattr(config, 'locations', None)
    if isinstance(locs, dict):
        for v in locs.values():
            if not v:
                continue
            try:
                dirs.append(Path(v))
            except Exception:
                pass

    # 3) settings may contain data/work directories
    s = getattr(config, 'settings', None)
    if s:
        for attr in ('work_directory', 'data_directory'):
            v = getattr(s, attr, None)
            if v:
                try:
                    dirs.append(Path(v))
                except Exception:
                    pass

    # 4) Default calibration output directory used by the `instamatic.calibrate_*` tools:
    #    %AppData%/calib  (Windows) or $AppData/calib (Unix)
    for env in ('AppData', 'APPDATA'):
        v = os.environ.get(env)
        if v:
            try:
                dirs.append(Path(v) / 'calib')
            except Exception:
                pass

    # Also try <instamatic base>/../calib if base_drc exists
    try:
        base = getattr(config, 'base_drc', None)
        if base:
            basep = Path(base)
            dirs.append(basep / 'calib')
            dirs.append(basep.parent / 'calib')
    except Exception:
        pass

    # 5) Nearby to this module (site-packages installs)
    try:
        here = Path(__file__).resolve()
        dirs.extend([here.parent, here.parent.parent, here.parent.parent.parent])
    except Exception:
        pass

    # 6) Current working directory (fallback)
    try:
        dirs.append(Path.cwd())
    except Exception:
        pass

    # De-duplicate while preserving order
    out: list[Path] = []
    seen = set()
    for d in dirs:
        try:
            d = d.expanduser()
        except Exception:
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _parse_stage_rotation_calib_text(text: str) -> dict:
    """Very small YAML-ish parser for the stage rotation calib files.

    This is used as a fallback when PyYAML is not available. It is
    intentionally minimal and only extracts the keys we care about.
    """

    # Match both top-level and nested (e.g. under `params:`) keys.
    out: dict = {}
    params: dict = {}
    speed_options: dict = {}

    def _maybe_float(v: str):
        try:
            return float(v)
        except Exception:
            return None

    for line in text.splitlines():
        # strip comments
        if '#' in line:
            line = line.split('#', 1)[0]
        if not line.strip():
            continue

        # Normalize
        key_val = line.split(':', 1)
        if len(key_val) != 2:
            continue
        k = key_val[0].strip()
        v = key_val[1].strip().strip('\'"')
        f = _maybe_float(v)

        if k in ('pace', 'windup', 'delay') and f is not None:
            # Could appear at top-level or inside `params:` in the fit file
            params[k] = f
        elif k in ('lower_lim', 'upper_lim') and f is not None:
            speed_options[k] = f

    if params:
        out['params'] = params
        # Also expose flat keys to support the other schema
        out.update(params)
    if speed_options:
        out['speed_options'] = speed_options

    return out


def load_stage_rotation_calibration(path_hint: str | None = None) -> StageRotationCalib | None:
    """Load stage rotation calibration YAML.

    Returns `StageRotationCalib` or None if no file could be loaded.

    On success, the returned object gets an extra attribute `_source_path` with
    the absolute path of the file that was loaded (for logging/debugging).
    """

    candidates: list[Path] = []
    if path_hint:
        candidates.append(Path(path_hint))
    for d in _candidate_calibration_dirs():
        for fn in STAGE_ROT_CALIB_FILENAMES:
            candidates.append(d / fn)

    for p in candidates:
        try:
            if not p.is_file():
                continue

            raw = p.read_text(encoding='utf-8', errors='ignore')
            if yaml is not None:
                d = yaml.safe_load(raw) or {}
            else:
                # Fallback: parse minimal keys
                d = _parse_stage_rotation_calib_text(raw)

            calib = StageRotationCalib.from_yaml_dict(d)
            # Attach source path for diagnostics
            try:
                setattr(calib, '_source_path', str(p.resolve()))
            except Exception:
                pass
            return calib
        except Exception:
            continue

    # No file found
    return None

    candidates: list[Path] = []
    if path_hint:
        candidates.append(Path(path_hint))
    for d in _candidate_calibration_dirs():
        for fn in STAGE_ROT_CALIB_FILENAMES:
            candidates.append(d / fn)

    for p in candidates:
        try:
            if p.is_file():
                with open(p, 'r', encoding='utf-8') as f:
                    d = yaml.safe_load(f) or {}
                calib = StageRotationCalib.from_yaml_dict(d)
                return calib
        except Exception:
            continue
    return None


def print_and_log(msg, logger=None):
    print(msg)
    if logger:
        logger.info(msg)


class Experiment(ExperimentBase):
    """Continuous rotation electron diffraction experiment (cRED-like).

    Key changes vs. the original experiment implementation:
      - Auto-stop in `footfree` mode when the target angle is reached.
      - Optionally pace acquisition with a minimum time interval between frames.
      - Uses ImgConversionDM (patched) for conversion/writing.

    Parameters (additional):
      footfree_speed:
          FEI/Tecnai stage speed fraction (0..1) used in 'footfree' mode.
          If your stage reaches the end angle too fast, lower this value.
      auto_stop_on_target:
          If True and mode=='footfree', stop acquisition automatically once the
          stage reaches `footfree_rotate_to`.
      frame_interval:
          Minimum seconds between diffraction frames (start-to-start).
          If None, defaults to exposure_time (i.e., no extra waiting beyond the camera).
    """

    def __init__(
        self,
        ctrl,
        path: str = None,
        log=None,
        flatfield: str = None,
        exposure_time: float = 0.5,
        unblank_beam: bool = False,
        mode: str = None,
        footfree_rotate_to: float = 60.0,
        enable_image_interval: bool = False,
        image_interval: int = 99999,
        diff_defocus: int = 0,
        exposure_time_image: float = 0.01,
        write_tiff: bool = True,
        write_xds: bool = True,
        write_dials: bool = True,
        write_red: bool = True,
        stop_event=None,
        # New (optional) parameters
        rotation_span_deg: Optional[float] = None,
        rotation_speed_deg_s: Optional[float] = None,
        use_stage_rotation_calibration: bool = True,
        stage_rotation_calibration_file: Optional[str] = None,
        footfree_speed: float = 1.0,
        auto_stop_on_target: bool = True,
        frame_interval: Optional[float] = None,
    ):
        super().__init__()
        self.ctrl = ctrl
        self.path = Path(path)
        self.exposure = float(exposure_time)
        self.unblank_beam = unblank_beam
        self.logger = log
        self.mode = mode
        # we need to be able to simulate using the serval camera as well - add "simulate" to part of the name.
        if 'simulate' in ctrl.cam.name:
            self.mode = 'simulate'
        self.stopEvent = stop_event or threading.Event()
        self.flatfield = flatfield

        self.footfree_rotate_to = float(footfree_rotate_to)
        # New: specify rotation relative span and/or requested speed in deg/s.
        # If rotation_span_deg is set, the target angle becomes start_angle + span.
        self.rotation_span_deg = None if rotation_span_deg is None else float(rotation_span_deg)
        self.rotation_speed_deg_s = (
            None if rotation_speed_deg_s is None else float(rotation_speed_deg_s)
        )
        self.use_stage_rotation_calibration = bool(use_stage_rotation_calibration)
        self.stage_rotation_calibration_file = stage_rotation_calibration_file
        self.stage_rotation_calib = load_stage_rotation_calibration(
            stage_rotation_calibration_file
        )
        self.stage_rotation_calib_path = getattr(
            self.stage_rotation_calib, '_source_path', None
        )
        self.rotation_speed_est_deg_s: Optional[float] = None
        self.rotation_speed_req_deg_s: Optional[float] = self.rotation_speed_deg_s

        self.footfree_speed = float(footfree_speed)
        self.auto_stop_on_target = bool(auto_stop_on_target)

        if frame_interval is None:
            self.frame_interval = self.exposure
        else:
            self.frame_interval = float(frame_interval)
            if self.frame_interval < self.exposure:
                # Cannot run faster than exposure; enforce a safe minimum.
                self.frame_interval = self.exposure

        self.diff_defocus = diff_defocus
        self.exposure_image = exposure_time_image

        self.write_tiff = write_tiff
        self.write_xds = write_xds
        self.write_dials = write_dials
        self.write_red = write_red
        self.write_pets = write_tiff  # TODO

        self.image_interval_enabled = enable_image_interval
        if enable_image_interval:
            self.image_interval = image_interval
            print_and_log(
                f'Image interval enabled: every {self.image_interval} frames an image with defocus {self.diff_defocus} will be displayed (t={self.exposure_image} s).',
                logger=self.logger,
            )
        else:
            self.image_interval = 99999

        self.relax_beam_before_experiment = (
            self.image_interval_enabled and config.settings.cred_relax_beam_before_experiment
        )

        self.track_stage_position = config.settings.cred_track_stage_positions
        self.stage_positions = []

        # Rotation bookkeeping (set during start_rotation)
        self._target_angle: Optional[float] = None
        self._rotation_direction: int = 0  # +1 or -1
        self._stop_reason: str = 'unknown'

        # Optional: VM server connection (guarded, because original code can crash if disabled)
        self.s2_c = 0
        if use_vm:
            try:
                self.s2 = socket.socket()
                vm_host = config.settings.VM_server_host
                vm_port = config.settings.VM_server_port
                self.s2.connect((vm_host, vm_port))
                print('VirtualBox server connected for autocRED.')
                self.s2_c = 1
            except BaseException:
                print('Is VM server running? Connection failed.')
                self.s2_c = 0

    def log_start_status(self):
        """Log the starting parameters."""
        self.now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f'Data recording started at: {self.now}')
        self.logger.info(f'Data collection exposure time: {self.exposure} s')
        self.logger.info(f'Data saving path: {self.path}')
        if self.mode == 'footfree':
            if self.rotation_span_deg is not None:
                self.logger.info(f'Requested rotation span: {self.rotation_span_deg} deg')
            if self.rotation_speed_req_deg_s is not None:
                self.logger.info(
                    f'Requested rotation speed: {self.rotation_speed_req_deg_s} deg/s'
                )
            self.logger.info(
                f'Use stage rotation calibration: {self.use_stage_rotation_calibration}'
            )
            if self.stage_rotation_calib_path:
                self.logger.info(
                    f'Stage rotation calibration file: {self.stage_rotation_calib_path}'
                )
            else:
                self.logger.info('Stage rotation calibration file: <not found>')
            self.logger.info(f'Fallback footfree speed setting (0..1): {self.footfree_speed}')

    def log_end_status(self):
        """Log the experimental values, write file `cRED_log.txt`"""
        start_xy = np.array(self.start_position[0:2])
        end_xy = np.array(self.end_position[0:2])

        print_and_log(
            f'Rotated {self.total_angle:.2f} degrees from {self.start_angle:.2f} to {self.end_angle:.2f} in {self.nframes} frames (step: {self.osc_angle:.4f})',
            logger=self.logger,
        )

        def fmt(arr):
            return f'[{arr[0]:.0f} {arr[1]:.0f}]'

        print_and_log(
            f'Stage moved from {fmt(start_xy)} to {fmt(end_xy)}, drift: {fmt(start_xy - end_xy)}',
            logger=self.logger,
        )
        self.logger.info(f'Start stage position: {self.start_position}')
        self.logger.info(f'End stage position: {self.end_position}')
        self.logger.info(f'Data collection camera length: {self.camera_length} mm')
        self.logger.info(f'Data collection spot size: {self.spotsize}')
        self.logger.info(f'Stop reason: {self._stop_reason}')

        self.logger.info(self.stage_positions)

        with open(self.path / 'cRED_log.txt', 'w') as f:
            print(f'Program: {instamatic.__long_title__}', file=f)
            print(f'Data Collection Time: {self.now}', file=f)
            print(f'Time Period Start: {self.t_start}', file=f)
            print(f'Time Period End: {self.t_end}', file=f)
            print(f'Starting angle: {self.start_angle:.2f} degrees', file=f)
            print(f'Ending angle: {self.end_angle:.2f} degrees', file=f)
            print(f'Rotation range: {self.end_angle - self.start_angle:.2f} degrees', file=f)
            print(f'Exposure Time: {self.exposure:.3f} s', file=f)
            if self.mode == 'footfree':
                print(f'Footfree speed setting (0..1): {self.footfree_speed:.4f}', file=f)
                if self.rotation_speed_req_deg_s is not None:
                    print(
                        f'Requested rotation speed: {self.rotation_speed_req_deg_s:.4f} deg/s',
                        file=f,
                    )
                if self.rotation_speed_est_deg_s is not None:
                    print(
                        f'Estimated rotation speed: {self.rotation_speed_est_deg_s:.4f} deg/s',
                        file=f,
                    )
            print(f'Acquisition time: {self.acquisition_time:.3f} s', file=f)
            print(f'Total time: {self.total_time:.3f} s', file=f)
            print(f'Spot Size: {self.spotsize}', file=f)
            print(f'Camera length: {self.camera_length} mm', file=f)
            print(f'Pixelsize: {self.pixelsize} px/Angstrom', file=f)
            print(f'Physical pixelsize: {self.physical_pixelsize} um', file=f)
            print(f'Wavelength: {self.wavelength} Angstrom', file=f)
            print(f'Stretch amplitude: {self.stretch_azimuth} %', file=f)
            print(f'Stretch azimuth: {self.stretch_amplitude} degrees', file=f)
            print(f'Rotation axis: {self.rotation_axis} radians', file=f)
            print(f'Oscillation angle: {self.osc_angle:.4f} degrees', file=f)
            print(f'Number of frames: {self.nframes_diff}', file=f)
            print(f'Stop reason: {self._stop_reason}', file=f)
            print(
                'Stage start: X {:6.0f} | Y {:6.0f} | Z {:6.0f} | A {:5.2f} | B {:5.2f}'.format(
                    *self.start_position
                ),
                file=f,
            )
            print(
                'Stage end:   X {:6.0f} | Y {:6.0f} | Z {:6.0f} | A {:5.2f} | B {:5.2f}'.format(
                    *self.end_position
                ),
                file=f,
            )

            if self.image_interval_enabled:
                print(
                    f'Image interval: every {self.image_interval} frames an image with defocus {self.diff_focus_defocused} (t={self.exposure_image} s).',
                    file=f,
                )
                print(f'Number of images: {self.nframes_image}', file=f)

            print('', file=f)
            print('References:', file=f)
            print(' -', instamatic.__citation__, file=f)
            print(' -', instamatic.__citation_cred__, file=f)

    def setup_paths(self):
        """Set up the paths for saving the data to."""
        print(f'\nOutput directory: {self.path}')
        self.tiff_path = self.path / 'tiff' if self.write_tiff else None
        self.smv_path = self.path / 'SMV' if (self.write_xds or self.write_dials) else None
        self.mrc_path = self.path / 'RED' if self.write_red else None

    def _init_footfree_target(self, start_angle: float, target_angle: float) -> None:
        self._target_angle = float(target_angle)
        self._rotation_direction = 1 if self._target_angle >= float(start_angle) else -1

    def _target_reached(self, current_angle: float) -> bool:
        if self._target_angle is None or self._rotation_direction == 0:
            return False
        if self._rotation_direction > 0:
            return current_angle >= (self._target_angle - END_ANGLE_TOLERANCE)
        else:
            return current_angle <= (self._target_angle + END_ANGLE_TOLERANCE)

    def start_rotation(self) -> float:
        """Controls the starting of the rotation of the experiment.

        In the default mode, wait for rotation to start (i.e. controlled
        via the pedals). In `footfree` mode, initialize rotation from
        current angle to target angle. In `simulate` mode, simulate the
        start condition.

        Returns the starting angle.
        """
        self.start_position = self.ctrl.stage.get()
        self.stage_positions.append((0, self.start_position))
        a = self.start_position[3]

        # Reset stop conditions
        self._target_angle = None
        self._rotation_direction = 0
        self._stop_reason = 'running'

        if self.mode == 'simulate':
            start_angle = a
            print('Data Recording started.')

        elif self.mode == 'footfree':
            start_angle = float(self.ctrl.stage.a)

            # Determine target angle:
            # - If rotation_span_deg is provided, rotate relative to the current angle.
            # - Else, keep the legacy absolute `footfree_rotate_to`.
            if self.rotation_span_deg is not None:
                rotate_to = start_angle + float(self.rotation_span_deg)
            else:
                rotate_to = float(self.footfree_rotate_to)

            # Determine footfree speed setting (0..1)
            speed_setting = float(self.footfree_speed)
            span_deg = abs(float(rotate_to) - float(start_angle))

            if self.rotation_speed_deg_s is not None:
                if self.use_stage_rotation_calibration and (
                    self.stage_rotation_calib is not None
                ):
                    try:
                        speed_setting = float(
                            self.stage_rotation_calib.speed_for_deg_per_s(
                                span_deg=span_deg,
                                target_deg_per_s=float(self.rotation_speed_deg_s),
                            )
                        )
                        self.rotation_speed_est_deg_s = float(
                            self.stage_rotation_calib.estimate_deg_per_s(
                                span_deg=span_deg, speed_setting=speed_setting
                            )
                        )
                        print_and_log(
                            f'Stage rotation: requested {self.rotation_speed_deg_s:.4f} deg/s, '
                            f'using footfree_speed={speed_setting:.4f} (estimated {self.rotation_speed_est_deg_s:.4f} deg/s) '
                            f'for span={span_deg:.2f} deg.',
                            logger=self.logger,
                        )
                    except Exception as e:
                        print_and_log(
                            f'Stage rotation calibration inversion failed ({e}); falling back to footfree_speed={speed_setting:.4f}.',
                            logger=self.logger,
                        )
                else:
                    print_and_log(
                        'Stage rotation calibration not available; '
                        f'cannot convert requested {self.rotation_speed_deg_s} deg/s to a footfree speed setting. '
                        f'Falling back to footfree_speed={speed_setting:.4f}.',
                        logger=self.logger,
                    )

            self.footfree_speed = float(speed_setting)
            self.footfree_rotate_to = float(rotate_to)

            self._init_footfree_target(start_angle=start_angle, target_angle=rotate_to)

            # Start rotation with an explicit speed if available (FEI/Tecnai uses 0..1)
            try:
                self.ctrl.stage.set_with_speed(
                    a=rotate_to, speed=self.footfree_speed, wait=False
                )
            except Exception:
                # Fallback: older API / non-FEI
                self.ctrl.stage.set(a=rotate_to, wait=False)

            # Wait until the stage actually starts moving (or it is already at target)
            a0 = start_angle
            t_wait0 = time.perf_counter()
            while not self.stopEvent.is_set():
                a_now = float(self.ctrl.stage.a)
                if abs(a_now - a0) >= ACTIVATION_THRESHOLD:
                    break
                if abs(a_now - rotate_to) <= END_ANGLE_TOLERANCE:
                    break
                if (time.perf_counter() - t_wait0) > START_MOVE_TIMEOUT:
                    break
                time.sleep(0.01)

            print('Data Recording started.')
            start_angle = float(self.ctrl.stage.a)

        else:
            print('Waiting for rotation to start...', end=' ')
            a0 = a
            while abs(a - a0) < ACTIVATION_THRESHOLD:
                if self.stopEvent.is_set():
                    break
                a = float(self.ctrl.stage.a)
            print('Data Recording started.')
            start_angle = float(a)

        if self.unblank_beam:
            print('Unblanking beam')
            self.ctrl.beam.unblank()

        return start_angle

    def relax_beam(self, n_cycles: int = 5):
        """Relax the beam prior to the experiment by toggling between the
        defocused/focused states."""
        print(f'Relaxing beam ({n_cycles} cycles)', end='')

        for _i in range(n_cycles):
            self.ctrl.difffocus.set(self.diff_focus_defocused)
            time.sleep(0.5)
            print('.', end='')
            self.ctrl.difffocus.set(self.diff_focus_proper)
            time.sleep(0.5)
            print('.', end='')

        print('Done.')

    def start_collection(self) -> bool:
        """Main experimental function, returns True if experiment runs
        normally, False if it is interrupted for whatever reason."""
        self.setup_paths()
        self.log_start_status()

        buffer = []
        image_buffer = []

        if self.ctrl.mode != 'diff':
            self.ctrl.mode.set('diff')

        self.diff_focus_proper = self.ctrl.difffocus.value
        self.diff_focus_defocused = self.diff_defocus + self.diff_focus_proper
        exposure_image = self.exposure_image

        if self.relax_beam_before_experiment:
            self.relax_beam()

        # start_rotation blocks until movement is detected (simulate/default), or until movement begins (footfree)
        self.start_angle = self.start_rotation()

        # Block camera notifications during acquisition
        self.ctrl.cam.block()

        i = 1
        t0 = time.perf_counter()
        next_frame_time = t0  # pacing for diffraction frames

        # For stall detection (footfree auto-stop)
        prev_angle = float(self.ctrl.stage.a)
        last_movement_t = time.perf_counter()

        while not self.stopEvent.is_set():
            # --- Auto-stop (footfree) ---
            if self.mode == 'footfree' and self.auto_stop_on_target:
                a_now = float(self.ctrl.stage.a)
                if abs(a_now - prev_angle) > ANGLE_STALL_EPS:
                    last_movement_t = time.perf_counter()
                    prev_angle = a_now
                try:
                    is_moving = bool(self.ctrl.stage.is_moving())
                except BaseException:
                    is_moving = True  # if unknown, do not auto-stop on stall

                if self._target_reached(a_now):
                    self._stop_reason = 'target_reached'
                    break

                # If the stage is no longer moving, stop to avoid collecting frames at the end angle
                if (
                    (not is_moving)
                    and (time.perf_counter() - last_movement_t) > STAGE_STALL_TIMEOUT
                    and i > 1
                ):
                    if self._target_angle is not None and abs(a_now - self._target_angle) <= (
                        END_ANGLE_TOLERANCE * 2
                    ):
                        self._stop_reason = 'target_reached'
                    else:
                        self._stop_reason = 'stage_stalled_or_stopped'
                    break

            # --- Image-interval branch (defocused tracking images) ---
            if i % self.image_interval == 0:
                t_start = time.perf_counter()
                acquisition_time = (t_start - t0) / max(i - 1, 1)

                self.ctrl.difffocus.set(self.diff_focus_defocused, confirm_mode=False)
                img, h = self.ctrl.get_image(exposure_image, header_keys=None)
                self.ctrl.difffocus.set(self.diff_focus_proper, confirm_mode=False)

                image_buffer.append((i, img, h))

                next_interval = t_start + acquisition_time

                while time.perf_counter() > next_interval:
                    next_interval += acquisition_time
                    i += 1

                diff = next_interval - time.perf_counter()  # seconds

                if self.track_stage_position and diff > 0.1:
                    self.stage_positions.append((i, self.ctrl.stage.get()))

                if diff > 0:
                    time.sleep(diff)

            # --- Normal diffraction frame branch ---
            else:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)

                img, h = self.ctrl.get_image(self.exposure, header_keys=None)
                buffer.append((i, img, h))

                # Enforce a minimum frame interval (start-to-start pacing)
                next_frame_time = time.perf_counter() + self.frame_interval

            i += 1

        t1 = time.perf_counter()

        # Stage handling
        if self.mode == 'footfree':
            if self._stop_reason not in ('target_reached',):
                # External stop -> stop stage preemptively
                try:
                    self.ctrl.stage.stop()
                except BaseException:
                    pass
            else:
                # Let the stage finish/settle at the target
                try:
                    self.ctrl.stage.wait()
                except BaseException:
                    pass

        # Clear external stop for next run
        self.stopEvent.clear()

        # Unblock camera notifications
        self.ctrl.cam.unblock()

        if self.mode == 'simulate':
            # simulate somewhat realistic end numbers
            self.ctrl.stage.x += np.random.randint(-5000, 5000)
            self.ctrl.stage.y += np.random.randint(-5000, 5000)
            self.ctrl.stage.a += np.random.randint(-100, 100)
            self.ctrl.magnification.set(300)

        self.end_position = self.ctrl.stage.get()
        self.end_angle = self.end_position[3]
        self.camera_length = int(self.ctrl.magnification.get())
        self.stage_positions.append((99999, self.end_position))

        try:
            is_moving = bool(self.ctrl.stage.is_moving())
        except BaseException:
            is_moving = False
        self.logger.info(f'Experiment finished, stage is moving: {is_moving}')

        if self.unblank_beam:
            print('Blanking beam')
            self.ctrl.beam.blank()

        # in case something went wrong starting data collection, return gracefully
        if i == 1:
            print_and_log('Data collection interrupted', logger=self.logger)
            return False

        self.spotsize = self.ctrl.spotsize
        self.nframes = i - 1  # len(buffer) can lie in case of frame skipping
        self.osc_angle = abs(self.end_angle - self.start_angle) / max(self.nframes, 1)
        self.t_start = t0
        self.t_end = t1
        self.total_time = t1 - t0
        self.acquisition_time = self.total_time / max(self.nframes, 1)
        self.total_angle = abs(self.end_angle - self.start_angle)
        self.rotation_axis = config.camera.camera_rotation_vs_stage_xy

        # Calibration values needed for SMV/MRC conversion. These may be absent in some installs;
        # do not abort the whole experiment if they are missing (TIFF-only should still work).
        try:
            self.pixelsize = config.calibration['diff']['pixelsize'][
                self.camera_length
            ]  # px / Angstrom
        except Exception as e:
            self.pixelsize = None
            print_and_log(
                f'Warning: pixelsize calibration missing for camera_length={self.camera_length}: {e}',
                logger=self.logger,
            )

        self.physical_pixelsize = getattr(
            getattr(config, 'camera', object()), 'physical_pixelsize', None
        )
        self.wavelength = getattr(getattr(config, 'microscope', object()), 'wavelength', None)
        self.stretch_azimuth = getattr(
            getattr(config, 'camera', object()), 'stretch_azimuth', 0.0
        )
        self.stretch_amplitude = getattr(
            getattr(config, 'camera', object()), 'stretch_amplitude', 0.0
        )

        self.nframes_diff = len(buffer)
        self.nframes_image = len(image_buffer)

        self.log_end_status()
        if self.nframes <= 3:
            print_and_log(
                f'Warning: only {self.nframes} frames collected. Writing available frames anyway.',
                logger=self.logger,
            )

        self.write_data(buffer)
        self.write_image_data(image_buffer)

        print('Data Collection and Conversion Done.')

        pathsmv_str = str(self.smv_path)
        msg = {
            'path': pathsmv_str,
            'rotrange': self.total_angle,
            'nframes': self.nframes,
            'osc': self.osc_angle,
        }
        msg_tosend = json.dumps(msg).encode('utf8')

        if getattr(self, 's2_c', 0):
            try:
                self.s2.send(msg_tosend)
                print('SMVs sent to XDS for processing.')
            except BaseException:
                pass

        return True

    def _write_tiff_direct(self, buffer: list, outdir: Optional[Path] = None) -> None:
        """Write TIFF files directly from the raw acquisition buffer.

        This path does **not** require camera-length calibration
        (pixelsize/wavelength), and is therefore a robust fallback for
        quick tests or partially configured installs.
        """
        if not buffer:
            return

        if outdir is None:
            outdir = self.tiff_path or (self.path / 'tiff')

        outdir.mkdir(exist_ok=True, parents=True)

        for i, img, h in buffer:
            fn = outdir / f'{i:05d}.tiff'
            write_tiff(str(fn), img, header=h)

        print_and_log(f'Wrote {len(buffer)} TIFF frames to: {outdir}', logger=self.logger)

    def write_data(self, buffer: list):
        """Write diffraction data in the buffer.

        - If the user selected **TIFF-only**, write TIFFs directly without conversion.
        - Otherwise, try the configured ImgConversion backend.
        - On any conversion/write error, fall back to TIFF-only (if enabled).

        This ensures the experiment always produces files when at least one frame was acquired.
        """
        if not buffer:
            print_and_log(
                'No diffraction frames were collected; nothing to write.', logger=self.logger
            )
            return

        tiff_only = bool(self.write_tiff) and not (
            self.write_xds or self.write_dials or self.write_red
        )
        if tiff_only:
            self._write_tiff_direct(buffer)
            return

        # Instantiate converter (copy buffer in case the implementation mutates it)
        try:
            img_conv = ImgConversion(
                buffer=list(buffer),
                osc_angle=self.osc_angle,
                start_angle=self.start_angle,
                end_angle=self.end_angle,
                rotation_axis=self.rotation_axis,
                acquisition_time=self.acquisition_time,
                flatfield=self.flatfield,
                pixelsize=self.pixelsize,
                physical_pixelsize=self.physical_pixelsize,
                wavelength=self.wavelength,
                stretch_amplitude=self.stretch_amplitude,
                stretch_azimuth=self.stretch_azimuth,
            )
        except Exception as e:
            print_and_log(f'ImgConversion init failed: {e}', logger=self.logger)
            if self.write_tiff:
                print_and_log('Falling back to TIFF-only write...', logger=self.logger)
                self._write_tiff_direct(buffer)
            return

        # Write image files (TIFF/SMV/MRC)
        print('Writing data files...')
        try:
            img_conv.threadpoolwriter(
                tiff_path=self.tiff_path if self.write_tiff else None,
                mrc_path=self.mrc_path if self.write_red else None,
                smv_path=self.smv_path if (self.write_xds or self.write_dials) else None,
                workers=8,
            )
        except Exception as e:
            print_and_log(f'Writing data via ImgConversion failed: {e}', logger=self.logger)
            if self.write_tiff:
                print_and_log('Falling back to TIFF-only write...', logger=self.logger)
                self._write_tiff_direct(buffer)
            return

        # Write helper/input files (best-effort)
        print('Writing input files...')
        try:
            if self.write_dials and self.smv_path is not None:
                img_conv.to_dials(self.smv_path)
            if self.write_red and self.mrc_path is not None:
                img_conv.write_ed3d(self.mrc_path)
            if (self.write_xds or self.write_dials) and self.smv_path is not None:
                img_conv.write_xds_inp(self.smv_path)
            if self.write_pets:
                img_conv.write_pets_inp(self.path)
            img_conv.write_beam_centers(self.path)
        except Exception as e:
            print_and_log(
                f'Post-processing file writing failed (continuing): {e}', logger=self.logger
            )

    def write_image_data(self, buffer: list):
        """Write image data in the buffer (defocused tracking images)."""
        if buffer:
            drc = self.path / 'tiff_image'
            drc.mkdir(exist_ok=True)
            while buffer:
                i, img, h = buffer.pop(0)
                fn = drc / f'{i:05d}.tiff'
                write_tiff(fn, img, header=h)
