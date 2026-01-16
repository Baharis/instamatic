from __future__ import annotations

import atexit
import json
import logging
import math
import socket
import threading
from io import BytesIO
from itertools import batched
from typing import Generator, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import tifffile
from serval_toolkit.camera import Camera as ServalCamera
from typing_extensions import Self

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
        host = http_url.hostname
        port = (http_url.port or 8080) + 1

        self.conn.measurement_stop()
        previous_config = self.conn.detector_config
        previous_destination = self.conn.destination

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('0.0.0.0', port))
            listener.listen(1)
            new_destination = {
                'Base': f'tcp://connect@{host}:{port}',
                'Format': 'jsonimage',
                'Mode': 'count',
            }
            self.conn.destination = {
                'Image': [
                    new_destination,
                ]
            }
            self.conn.set_detector_config(
                TriggerMode=mode,
                ExposureTime=exposure,
                TriggerPeriod=exposure + self.dead_time,
                nTriggers=n_frames,
            )

            trigger_thread = threading.Thread(target=self.conn.measurement_start)
            trigger_thread.start()
            try:
                sock, addr = listener.accept()
            except socket.timeout:
                raise TimeoutError('Serval failed to connect back within 60 seconds.')
            with sock:
                yield from ServalMovieDeserializer(sock, n_frames)
            trigger_thread.join(timeout=2.0)

        finally:
            listener.close()
            try:
                self.conn.measurement_stop()
            except Exception as e:
                logger.error(f'Error stopping measurement: {e}')
            try:
                self.conn.destination = previous_destination
                self.conn.set_detector_config(**previous_config)
            except Exception as e:
                logger.error(f'Error restoring config: {e}')

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


class ServalMovieDeserializer(Iterator[np.ndarray]):
    """Deserializes Serval camera TCP byte stream from socket into images."""

    def __init__(self, sock: socket.socket, n_frames: int):
        self.sock = sock
        self.buffer = bytearray(65535)
        self.buffer_size = 0
        self.offset = 0
        self.i_frame = 0
        self.n_frames = n_frames

        self.width: int = 0
        self.height: int = 0
        self.depth: int = 0
        self.size: int = 0
        self.dtype: np.dtype = np.uint32

    def find_header_size(self) -> int:
        """Find the index of curly bracket that closes current header + 1."""
        bracket_depth: int = 1
        header_end_idx: int = -1
        next_closing_idx: int = -1
        scanning_idx: int = self.offset + 1

        while header_end_idx < 0:
            if next_closing_idx < scanning_idx:
                try:
                    next_closing_idx = self.buffer.index(b'}', scanning_idx)
                except ValueError:
                    return -1
            try:
                next_opening_idx = self.buffer.index(b'{', scanning_idx, next_closing_idx)
            except ValueError:
                bracket_depth -= 1
                if bracket_depth == 0:
                    return next_closing_idx + 1
                scanning_idx = next_closing_idx + 1
            else:
                bracket_depth += 1
                scanning_idx = next_opening_idx + 1
        return -1

    def reconfigure_from_header(self, header_size: int) -> None:
        json_bytes = self.buffer[self.offset : header_size]
        header_str = json_bytes.decode('utf-8')
        header_dict = json.loads(header_str)
        self.width = w = header_dict['width']
        self.height = h = header_dict['height']
        self.depth = d = header_dict['bitDepth'] // 8
        self.size = header_dict.get('dataSize', w * h * d)
        self.dtype = np.uint32 if d > 2 else np.uint16 if d > 1 else np.uint8
        self.buffer += bytearray(self.n_frames * (header_size + self.size))

    def __next__(self) -> np.ndarray:
        if self.i_frame >= self.n_frames:
            raise StopIteration
        header_size = self.find_header_size()
        while header_size < 0:
            self.buffer_size += self.sock.recv_into(memoryview(self.buffer)[self.buffer_size :])
            header_size = self.find_header_size()
        if self.i_frame == 0:
            self.reconfigure_from_header(header_size)
        image_start = header_size
        image_end = image_start + self.size
        while self.buffer_size < image_end:
            self.buffer_size += self.sock.recv_into(memoryview(self.buffer)[self.buffer_size :])
        image_view = memoryview(self.buffer)[image_start:image_end]
        img_array = np.frombuffer(image_view, dtype=self.dtype)
        image = img_array.reshape((self.height, self.width))
        self.offset = image_end
        self.i_frame += 1
        return image


if __name__ == '__main__':
    cam = CameraServal()
    from IPython import embed

    embed()
