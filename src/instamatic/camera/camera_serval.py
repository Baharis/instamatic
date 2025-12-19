from __future__ import annotations

import atexit
import json
import logging
import math
import socket
import time
from io import BytesIO
from itertools import batched
from typing import Generator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import tifffile
from serval_toolkit.camera import Camera as ServalCamera

from instamatic.camera.camera_base import CameraBase

logger = logging.getLogger(__name__)

# Start servers in serval_toolkit:
# 1. `java -jar .\emu\tpx3_emu.jar`
# 2. `java -jar .\server\serv-2.1.3.jar`
# 3. launch `instamatic`

Ignore = object()  # sentinel object: informs `_get_images` to get a single image


class CameraServal(CameraBase):
    """Interfaces with Serval from ASI."""

    streamable = True
    MIN_EXPOSURE = 0.000001
    MAX_EXPOSURE = 10.0
    BAD_EXPOSURE_MSG = 'Requested exposure exceeds native Serval support (>0-10s)'
    TCP_CHUNK_SIZE = 4096
    TCP_PORT_OFFSET = +1  # TCP PORT = HTTP_PORT + TCP_PORT_OFFSET

    def __init__(self, name='serval'):
        """Initialize camera module."""
        super().__init__(name)
        self.establish_connection()
        self.dead_time = (
            self.detector_config['TriggerPeriod'] - self.detector_config['ExposureTime']
        )
        logger.info(f'Camera {self.get_name()} initialized')
        atexit.register(self.release_connection)

    def get_image(self, exposure: Optional[float] = None, **kwargs) -> np.ndarray:
        """Image acquisition interface. If the exposure is not given, the
        default value is read from the config file. Binning is ignored.

        exposure: `float` or `None`
            Exposure time in seconds.
        """
        return self._get_images(n_frames=Ignore, exposure=exposure, **kwargs)

    def _get_images(
        self,
        n_frames: Union[int, Ignore],
        exposure: Optional[float] = None,
        **kwargs,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """General media acquisition dispatcher for other protected methods."""
        n: int = 1 if n_frames is Ignore else n_frames
        e: float = self.default_exposure if exposure is None else exposure

        if n_frames == 0:  # single image is communicated via n_frames = Ignore
            return []

        elif e < self.MIN_EXPOSURE:
            logger.warning('%s: %d', self.BAD_EXPOSURE_MSG, e)
            if n_frames is Ignore:
                return self._get_image_null(exposure=e, **kwargs)
            return [self._get_image_null(exposure=e, **kwargs) for _ in range(n)]

        elif e > self.MAX_EXPOSURE:
            logger.warning('%s: %d', self.BAD_EXPOSURE_MSG, e)
            n1 = math.ceil(e / self.MAX_EXPOSURE)
            e = (e + self.dead_time) / n1 - self.dead_time
            images = self._get_image_stack(n_frames=n * n1, exposure=e, **kwargs)
            if n_frames is Ignore:
                return self._spliced_sum(images, exposure=e)
            return [self._spliced_sum(i, exposure=e) for i in batched(images, n1)]

        else:  # if exposure is within limits
            if n_frames is Ignore:
                return self._get_image_single(exposure=e, **kwargs)
            return self._get_image_stack(n_frames=n, exposure=e, **kwargs)

    def _spliced_sum(self, arrays: Sequence[np.ndarray], exposure: float) -> np.ndarray:
        """Sum a series of arrays while applying a dead time correction."""
        array_sum = sum(arrays, np.zeros_like(arrays[0]))
        total_exposure = len(arrays) * exposure + (len(arrays) - 1) * self.dead_time
        live_fraction = len(arrays) * exposure / total_exposure
        return (array_sum / live_fraction).astype(arrays[0].dtype)

    def _get_image_null(self, **_) -> np.ndarray:
        logger.debug('Creating a synthetic image with zero counts')
        return np.zeros(shape=self.get_image_dimensions(), dtype=np.int32)

    def _get_image_single(self, exposure: float, **_) -> np.ndarray:
        """Request a single frame in the mode in a trigger collection mode."""
        logger.debug(f'Collecting a single image with exposure {exposure} s')
        # Upload exposure settings (Note: will do nothing if no change in settings)
        self.conn.set_detector_config(
            ExposureTime=exposure,
            TriggerPeriod=exposure + self.dead_time,
        )

        # Check if measurement is running. If not: start
        db = self.conn.dashboard
        if db['Measurement'] is None or db['Measurement']['Status'] != 'DA_RECORDING':
            self.conn.measurement_start()

        # Start the acquisition
        self.conn.trigger_start()

        # Request a frame. Will be streamed *after* the exposure finishes
        response = self.conn.get_request('/measurement/image')
        return tifffile.imread(BytesIO(response.content))

    def _get_image_stack(self, n_frames: int, exposure: float, **_) -> list[np.ndarray]:
        """Get a series of images in a mode with minimal dead time."""
        return list(self.get_movie(n_frames=n_frames, exposure=exposure))

    def get_movie(
        self, n_frames: int, exposure: Optional[float] = None, **kwargs
    ) -> Generator[np.ndarray, None, None]:
        """Yield `n_frames` images received via a TCP stream with minimal dead
        time. If the exposure is not given, the default value is read from the
        config file. Binning is ignored.

        n_frames: `int`
            Number of frames to collect
        exposure: `float` or `None`
            Exposure time in seconds.
        """
        logger.debug(f'Collecting {n_frames}-frame movie with exposure {exposure} s via TCP')
        mode: str = 'AUTOTRIGSTART_TIMERSTOP' if self.dead_time else 'CONTINUOUS'
        exposure: float = self.default_exposure if exposure is None else exposure

        http_url = urlparse(self.conn.url)
        tcp_host = http_url.hostname
        tcp_port = (http_url.port or 8080) + self.TCP_PORT_OFFSET

        self.conn.measurement_stop()
        previous_config = self.conn.detector_config
        previous_destination = self.conn.destination

        try:
            self.conn.set_detector_config(
                TriggerMode=mode,
                ExposureTime=exposure,
                TriggerPeriod=exposure + self.dead_time,
                nTriggers=n_frames,
            )
            self.conn.destination = {  # listen mode: serval waits for us to connect
                'Image': [
                    {
                        'Base': f'tcp://listen@0.0.0.0:{tcp_port}',
                        'Format': 'jsonimage',
                        'Mode': 'count',
                    }
                ]
            }

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(max(2.0, (exposure + self.dead_time) * 5))
            for attempt in range(10):
                try:
                    sock.connect((tcp_host, tcp_port))
                    break
                except ConnectionRefusedError:
                    if attempt == 9:
                        raise
                    time.sleep(0.05)

            with sock:
                self.conn.measurement_start()
                yield from self._tcp_stream(sock, n_frames)

        finally:
            try:
                self.conn.measurement_stop()
            except Exception as e:
                logger.error(f'Error stopping measurement: {e}')
            try:
                self.conn.destination = previous_destination
                self.conn.set_detector_config(**previous_config)
            except Exception as e:
                logger.error(f'Error restoring config: {e}')

    def _tcp_stream(self, sock: socket.socket, n_frames: int) -> Generator[np.ndarray]:
        """Parse 'jsonimage', yield images from a raw data via TCP socket."""
        buffer = bytearray()
        frames_yielded = 0

        while frames_yielded < n_frames:  # Read until enough to identify the JSON header
            chunk = sock.recv(self.TCP_CHUNK_SIZE)
            if not chunk:
                break
            buffer.extend(chunk)

            while True:
                if not buffer:
                    break

                # Scanning json-image for {}-delimited JSON header
                brace_depth: int = 0
                header_start_idx: int = -1
                header_end_idx: int = -1
                next_closing_idx: int = -1
                scanning_idx = 0
                while header_end_idx < 0:
                    if next_closing_idx < scanning_idx:
                        try:
                            next_closing_idx = buffer.index(b'}', scanning_idx)
                        except ValueError:
                            break  # read data until a closing bracket is found
                    try:
                        next_opening_idx = buffer.index(b'{', scanning_idx, next_closing_idx)
                        if brace_depth == 0:
                            header_start_idx = next_opening_idx
                    except ValueError:
                        brace_depth -= 1
                        scanning_idx = next_closing_idx + 1
                        if brace_depth == 0:
                            header_end_idx = next_closing_idx + 1
                    else:
                        brace_depth += 1
                        scanning_idx = next_opening_idx + 1
                if header_end_idx == -1:
                    break  # propagate break to read more

                # Reading in and parsing information in the json header
                json_bytes = buffer[header_start_idx:header_end_idx]
                header_str = json_bytes.decode('utf-8')
                header_dict = json.loads(header_str)
                width_default, height_default = self.get_image_dimensions()
                width = header_dict.get('width', width_default)
                height = header_dict.get('height', height_default)
                bit_depth = header_dict.get('bitDepth', 16)
                data_size = header_dict.get('dataSize', width * height * bit_depth // 8)
                del buffer[:header_end_idx]

                # Reading in missing data
                if len(buffer) < data_size:
                    while (missing := data_size - len(buffer)) > 0:
                        chunk = sock.recv(min(missing, 65536))  # read large chunks
                        if not chunk:
                            raise ConnectionError('Socket closed mid-frame')
                        buffer.extend(chunk)
                dt = np.uint32 if bit_depth > 16 else np.uint16 if bit_depth > 8 else np.uint8
                img_array = np.frombuffer(buffer[:data_size], dtype=dt)
                shape = (height, width) if width and height else self.get_image_dimensions()
                yield img_array.reshape(shape)
                frames_yielded += 1
                del buffer[:data_size]
                if frames_yielded >= n_frames:
                    return

    def get_image_dimensions(self) -> Tuple[int, int]:
        """Get the binned dimensions reported by the camera."""
        binning = self.get_binning()
        dim_x, dim_y = self.get_camera_dimensions()

        dim_x = int(dim_x / binning)
        dim_y = int(dim_y / binning)

        return dim_x, dim_y

    def establish_connection(self) -> None:
        """Establish connection to the camera."""
        self.conn = ServalCamera()
        self.conn.connect(self.url)
        self.conn.set_chip_config_files(
            bpc_file_path=self.bpc_file_path, dacs_file_path=self.dacs_file_path
        )
        self.conn.set_detector_config(**self.detector_config)

        img_dest = {'Base': 'http://localhost', 'Format': 'tiff', 'Mode': 'count'}
        self.conn.destination = {'Image': [img_dest]}

    def release_connection(self) -> None:
        """Release the connection to the camera."""
        self.conn.measurement_stop()
        name = self.get_name()
        msg = f"Connection to camera '{name}' released"
        logger.info(msg)


if __name__ == '__main__':
    cam = CameraServal()
    from IPython import embed

    embed()
