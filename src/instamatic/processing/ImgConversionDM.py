from __future__ import annotations

from .ImgConversion import *


class ImgConversionDM(ImgConversion):
    """This class is for converting data collected with the insteaDMatic DM-
    script Files can be generated for REDp, DIALS, XDS, and PETS.

    The image buffer is passed as a list of tuples, where each tuple
    contains the index (int), image data (2D numpy array),
    metadata/header (dict). The buffer index must start at 1.
    """

    def __init__(
        self,
        buffer: list,  # image buffer, list of (index [int], image data [2D numpy array], header [dict])
        osc_angle: float,  # degrees, oscillation angle of the rotation
        start_angle: float,  # degrees, start angle of the rotation
        end_angle: float,  # degrees, end angle of the rotation
        rotation_axis: float,  # radians, specifies the position of the rotation axis
        acquisition_time: float,  # seconds, acquisition time (exposure time + overhead)
        flatfield: str = 'flatfield.tiff',
        pixelsize: float = None,  # p/Angstrom, size of the pixels (overrides camera_length)
        physical_pixelsize: float = None,  # mm, physical size of the pixels (overrides camera length)
        wavelength: float = None,  # Angstrom, relativistic wavelength of the electron beam
        stretch_amplitude: float = 0.0,  # ignored (compat)
        stretch_azimuth: float = 0.0,  # ignored (compat)
        method: str = 'continuous-rotation 3D ED',  # or 'stills' or 'precession', used for CIF/documentation
    ):
        if flatfield is not None:
            try:
                flatfield, _h = read_tiff(flatfield)
            except Exception as e:
                logger.warning(
                    f'Flatfield file could not be read ({flatfield}); continuing without flatfield. Error: {e}'
                )
                flatfield = None
        self.flatfield = flatfield

        self.headers = {}
        self.data = {}

        self.smv_subdrc = 'data'

        if not buffer:
            raise ValueError(
                'ImgConversionDM received an empty image buffer; nothing to convert/save.'
            )

        # Do not mutate the caller's buffer (Experiment may want to inspect it after conversion)
        data_shape = None
        for i, img, h in list(buffer):
            self.headers[i] = h

            if self.flatfield is not None:
                self.data[i] = apply_flatfield_correction(img, self.flatfield)
            else:
                self.data[i] = img

            if data_shape is None:
                data_shape = img.shape

        self.observed_range = set(self.data.keys())
        self.complete_range = set(range(min(self.observed_range), max(self.observed_range) + 1))
        self.missing_range = self.observed_range ^ self.complete_range

        self.data_shape = data_shape

        self.pixelsize = pixelsize
        self.physical_pixelsize = physical_pixelsize
        self.wavelength = wavelength

        self.use_beamstop = False
        self.mean_beam_center, self.beam_center_std = self.get_beam_centers()

        if self.wavelength is None or self.physical_pixelsize is None or self.pixelsize is None:
            raise ValueError(
                f'Missing required calibration for distance calculation: '
                f'wavelength={self.wavelength}, physical_pixelsize={self.physical_pixelsize}, pixelsize={self.pixelsize}. '
                'Provide these in config (microscope.wavelength, camera.physical_pixelsize, calibration.diff.pixelsize) '
                'or pass them explicitly.'
            )
        self.distance = (1 / self.wavelength) * (self.physical_pixelsize / self.pixelsize)
        self.osc_angle = osc_angle
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.rotation_axis = rotation_axis

        self.acquisition_time = acquisition_time
        self.rotation_speed = 0  # n/a

        logger.debug(f'Primary beam at: {self.mean_beam_center}')

        self.name = 'DigitalMicrograph'
        self.method = method

        from .XDS_templateDM import XDS_template

        self.XDS_template = XDS_template

        self.check_settings()
