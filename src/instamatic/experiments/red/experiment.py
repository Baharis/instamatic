from __future__ import annotations

import datetime
import os
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from instamatic import config
from instamatic.experiments.experiment_base import ExperimentBase
from instamatic.formats import write_tiff
from instamatic.processing.ImgConversionDM import ImgConversionDM as ImgConversion


class Experiment(ExperimentBase):
    """Initialize stepwise rotation electron diffraction experiment.

    ctrl:
        Instance of instamatic.controller.TEMController
    path:
        `str` or `pathlib.Path` object giving the path to save data at
    log:
        Instance of `logging.Logger`
    flatfield:
        Path to flatfield correction image
    """

    def __init__(
        self,
        ctrl,
        path: str = None,
        log=None,
        flatfield=None,
        write_tiff: bool = True,
        write_red: bool = True,
    ):
        super().__init__()
        self.ctrl = ctrl
        self.path = Path(path)

        self.mrc_path = self.path / 'mrc'
        self.tiff_path = self.path / 'tiff'
        self.tiff_image_path = self.path / 'tiff_image'

        self.tiff_path.mkdir(exist_ok=True, parents=True)
        self.tiff_image_path.mkdir(exist_ok=True, parents=True)
        self.mrc_path.mkdir(exist_ok=True, parents=True)

        self.logger = log
        self.camtype = ctrl.cam.name

        self.flatfield = flatfield

        self.write_tiff = bool(write_tiff)
        self.write_red = bool(write_red)

        self.offset = 1
        self.current_angle = None
        self.buffer = []

    def start_collection(self, exposure_time: float, tilt_range: float, stepsize: float):
        """Start or continue data collection for `tilt_range` degrees with
        steps given by `stepsize`, To finalize data collection and write data
        files, run `self.finalize`.

        The number of images collected is defined by `tilt_range / stepsize`.

        exposure_time:
            Exposure time for each image in seconds
        tilt_range:
            Tilt range starting from the current angle in degrees. Must be positive.
        stepsize:
            Step size for the angle in degrees, controls the direction and can be positive or negative
        """
        self.spotsize = self.ctrl.spotsize
        self.now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('Data recording started at: {self.now}')
        self.logger.info(
            f'Exposure time: {exposure_time} s, Tilt range: {tilt_range}, step size: {stepsize}'
        )

        ctrl = self.ctrl

        if stepsize < 0:
            tilt_range = -abs(tilt_range)
        else:
            tilt_range = abs(tilt_range)

        if self.current_angle is None:
            self.start_angle = start_angle = ctrl.stage.a
        else:
            start_angle = self.current_angle + stepsize

        tilt_positions = np.arange(start_angle, start_angle + tilt_range, stepsize)
        print(f'\nStart_angle: {start_angle:.3f}')
        # print "Angles:", tilt_positions
        # Safety: block videostream before any SEMCCD image acquisition to avoid socket desync
        cam_blocked = False
        if getattr(ctrl.cam, 'streamable', False):
            try:
                ctrl.cam.block()
                cam_blocked = True
                time.sleep(
                    0.5
                )  # let stream thread stop / drain before starting RED acquisition
            except Exception:
                cam_blocked = False

        try:
            image_mode = ctrl.mode.get()
            if image_mode != 'diff':
                fn = self.tiff_image_path / f'image_{self.offset}.tiff'
                img, h = self.ctrl.get_image(exposure_time / 5)
                time.sleep(0.2)
                write_tiff(fn, img, header=h)
                ctrl.mode.set('diff')
                time.sleep(1.0)  # add some delay to account for beam lag

            # for i, a in enumerate(tilt_positions):
            for i, angle in enumerate(tqdm(tilt_positions)):
                ctrl.stage.a = angle

                j = i + self.offset

                img, h = self.ctrl.get_image(exposure_time)

                self.buffer.append((j, img, h))

            self.offset += len(tilt_positions)
            self.nframes = j

            self.end_angle = end_angle = ctrl.stage.a

            self.camera_length = int(self.ctrl.magnification.get())
            self.stepsize = stepsize
            self.exposure_time = exposure_time

            with open(self.path / 'summary.txt', 'a') as f:
                print(
                    f'{self.now}: Data collected from {start_angle:.2f} degree to {end_angle:.2f} degree in {len(tilt_positions)} frames.',
                    file=f,
                )
                print(
                    f'Data collected from {start_angle:.2f} degree to {end_angle:.2f} degree in {len(tilt_positions)} frames.'
                )

            self.logger.info(
                f'Data collected from {start_angle:.2f} degree to {end_angle:.2f} degree (camera length: {self.camera_length} mm).'
            )

            self.current_angle = angle
            print(f'Done, current angle = {self.current_angle:.2f} degrees')

            if image_mode != 'diff':
                ctrl.mode.set(image_mode)

        finally:
            if cam_blocked:
                try:
                    ctrl.cam.unblock()
                except Exception:
                    pass

    def finalize(self):
        """Finalize data collection after `self.start_collection` has been run.

        Write data in `self.buffer` to path given by `self.path`.
        """
        self.logger.info(f'Data saving path: {self.path}')
        self.rotation_axis = getattr(config.camera, 'camera_rotation_vs_stage_xy', 0.0)

        # If collection fails before camera_length is set, avoid crashing in finalize()
        if not hasattr(self, 'camera_length'):
            self.pixelsize = None
            if self.logger:
                self.logger.error(
                    'Finalize called but camera_length was never set (collection failed early).'
                )
            print(
                'WARNING: Collection failed before camera_length was set; skipping finalize().'
            )
            return

        try:
            self.pixelsize = config.calibration['diff']['pixelsize'][
                self.camera_length
            ]  # px / Angstrom
        except Exception as e:
            self.pixelsize = None
            if self.logger:
                self.logger.exception(
                    'Pixelsize calibration missing for camera_length=%s', self.camera_length
                )
            print(
                f'WARNING: pixelsize calibration missing for camera_length={self.camera_length}. ({e})'
            )
        self.physical_pixelsize = getattr(config.camera, 'physical_pixelsize', None)  # mm
        self.wavelength = getattr(config.microscope, 'wavelength', None)  # angstrom
        self.stretch_azimuth = getattr(config.camera, 'stretch_azimuth', 0.0)
        self.stretch_amplitude = getattr(config.camera, 'stretch_amplitude', 0.0)

        with open(self.path / 'summary.txt', 'a') as f:
            print(f'Rotation range: {self.end_angle - self.start_angle:.2f} degrees', file=f)
            print(f'Exposure Time: {self.exposure_time:.3f} s', file=f)
            print(f'Spot Size: {self.spotsize}', file=f)
            print(f'Camera length: {self.camera_length} mm', file=f)
            print(f'Pixelsize: {self.pixelsize} px/Angstrom', file=f)
            print(f'Physical pixelsize: {self.physical_pixelsize} um', file=f)
            print(f'Wavelength: {self.wavelength} Angstrom', file=f)
            print(f'Stretch amplitude: {self.stretch_amplitude} %', file=f)
            print(f'Stretch azimuth: {self.stretch_azimuth} degrees', file=f)
            print(f'Rotation axis: {self.rotation_axis} radians', file=f)
            print(f'Stepsize: {self.stepsize:.4f} degrees', file=f)
            print(f'Number of frames: {self.nframes}', file=f)

        # --- Log binning (binsize) information for all frames before conversion ---
        # Note: `ImgConversionDM` may consume/mutate the buffer depending on implementation,
        # so we log first and pass a shallow copy into the converter.
        binning_map = {}
        for frame_index, _img, header in self.buffer:
            binsize = None
            if isinstance(header, dict):
                binsize = (
                    header.get('ImageBinsize', None)
                    or header.get('binning', None)
                    or header.get('Binning', None)
                )
            binning_map[int(frame_index)] = binsize

        unique_bins = sorted({b for b in binning_map.values() if b is not None})
        if self.logger:
            self.logger.info(
                'Binning summary (unique ImageBinsize values): %s',
                unique_bins if unique_bins else 'N/A',
            )
        with open(self.path / 'summary.txt', 'a') as f:
            print(
                f'Binning summary (unique ImageBinsize values): {unique_bins if unique_bins else "N/A"}',
                file=f,
            )
            print(f'Binning map file: {self.path / "binning_map.txt"}', file=f)

        # Write a per-frame binning map to a separate log file (keeps the main logger concise)
        binning_log_fn = self.path / 'binning_map.txt'
        with open(binning_log_fn, 'w', encoding='utf-8') as f:
            print('# frame_index	ImageBinsize	tiff	mrc', file=f)
            for k in sorted(binning_map):
                tiff_fn = (self.tiff_path / f'{k:05d}.tiff') if self.write_tiff else ''
                mrc_fn = (self.mrc_path / f'{k:05d}.mrc') if self.write_red else ''
                print(f'{k:05d}	{binning_map[k]}	{tiff_fn}	{mrc_fn}', file=f)

        if self.logger:
            self.logger.info('Binning map written to: %s', binning_log_fn)

        buffer_copy = list(self.buffer)

        img_conv = ImgConversion(
            buffer=buffer_copy,
            osc_angle=self.stepsize,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            rotation_axis=self.rotation_axis,
            acquisition_time=self.exposure_time,
            flatfield=self.flatfield,
            pixelsize=self.pixelsize,
            physical_pixelsize=self.physical_pixelsize,
            wavelength=self.wavelength,
        )

        print('Writing data files...')
        img_conv.threadpoolwriter(
            tiff_path=self.tiff_path if self.write_tiff else None,
            mrc_path=self.mrc_path if self.write_red else None,
            workers=8,
        )

        print('Writing input files...')
        if self.write_red:
            img_conv.write_ed3d(self.mrc_path)
        else:
            if self.logger:
                self.logger.info(
                    'Skipping REDp/ED3D output (MRC) because it is disabled in the GUI.'
                )

        if self.write_tiff:
            img_conv.write_pets_inp(self.path)
        else:
            if self.logger:
                self.logger.info(
                    'Skipping PETS output (TIFF) because it is disabled in the GUI.'
                )

        img_conv.write_beam_centers(self.path)

        print('Data Collection and Conversion Done.')
        print()

        return True

    def teardown(self):
        self.finalize()


def main():
    from instamatic import controller

    ctrl = controller.initialize()

    import logging

    log = logging.getLogger(__name__)

    exposure_time = 0.5
    tilt_range = 10
    stepsize = 1.0

    i = 1
    while True:
        expdir = f'experiment_{i}'
        if os.path.exists(expdir):
            i += 1
        else:
            break

    print(f'\nData directory: {expdir}')

    red_exp = Experiment(ctrl=ctrl, path=expdir, log=log, flatfield=None)
    red_exp.start_collection(
        exposure_time=exposure_time, tilt_range=tilt_range, stepsize=stepsize
    )

    input('Press << Enter >> to start the experiment... ')

    while not input(
        f'\nPress << Enter >> to continue for another {tilt_range} degrees. [any key to finalize] '
    ):
        red_exp.start_collection(
            exposure_time=exposure_time, tilt_range=tilt_range, stepsize=stepsize
        )

    red_exp.finalize()


if __name__ == '__main__':
    main()
