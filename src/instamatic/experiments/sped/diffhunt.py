from __future__ import annotations

import multiprocessing as mp
import multiprocessing.shared_memory
import queue
import uuid

import numpy as np

from instamatic.experiments.sped.util import SPEDLoc

N_PROCESSORS = 4


mp.set_start_method('spawn', force=True)


class DiffHuntDispatcher:
    """Proxy class: ask workers on other processes if image has diffraction"""

    def __init__(self, shape, dtype):
        self.shape: tuple[int, int] = shape
        self.dtype: np.dtype = dtype

        self.buffer: np.ndarray = np.array([], dtype=dtype)
        self.buffer_name: str = ''
        self.buffer_ptr: int = 0

        self.queries = mp.Queue()
        self.answers = mp.Queue()
        self.workers: list[mp.Process] = self.initialize_workers()

    def initialize_workers(self) -> list[mp.Process]:
        """Run once at the start of experiment to spawn eval processes."""
        args = (self.queries, self.answers, self.dtype)
        workers = []
        for i in range(N_PROCESSORS):
            p = mp.Process(target=diff_hunt_worker, args=(i, *args), daemon=1)
            workers.append(p)
            p.start()
        return workers

    def switch_buffer(self, n_frames: int = 100, name: str = None) -> None:
        """Configure a new mp shared memory space to buffer a frame stack."""
        self.buffer_name = name if name is not None else 'SPED_' + uuid.uuid4().hex
        buffer_shape = (n_frames, self.shape[0], self.shape[1])
        size = np.prod(buffer_shape) * np.dtype(self.dtype).itemsize
        shm = mp.shared_memory.SharedMemory(name=self.buffer_name, create=True, size=size)
        self.buffer = np.ndarray(buffer_shape, dtype=self.dtype, buffer=shm.buf)
        self.buffer_ptr: int = 0

    def submit(self, frame: np.ndarray) -> None:
        """Request eval of 1 frame from buffer stored on some shared memory."""
        if self.buffer_ptr >= self.buffer.shape[0]:
            raise RuntimeError(f'{self.buffer_name} buffer overflow')
        self.buffer[self.buffer_ptr, :, :] = frame
        self.queries.put((self.buffer_name, self.buffer.shape, self.buffer_ptr))
        self.buffer_ptr += 1

    def poll(self):
        try:
            return self.answers.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.queries.put(None)
        for p in self.workers:
            p.join()


def diff_hunt_worker(
    worker_id: int,
    queries: mp.Queue,
    answers: mp.Queue,
    dtype: np.dtype,
):
    """Evaluates if shared frames have diffraction on a separate processor."""
    buffer: np.ndarray = np.array([], dtype=dtype)
    buffer_name: str = ''

    while True:
        q = queries.get(block=True)

        if q is None:
            queries.put(None)
            return

        assert isinstance(q, tuple) and len(q) == 3
        q_buffer_name, q_buffer_shape, q_buffer_ptr = q

        if q_buffer_name != buffer_name:
            buffer_name = q_buffer_name
            shm = mp.shared_memory.SharedMemory(name=buffer_name)
            buffer = np.ndarray(q_buffer_shape, dtype=dtype, buffer=shm.buf)

        frame = buffer[q_buffer_ptr]
        has_diffraction: bool = detect_diffraction(frame)
        answers.put((q_buffer_ptr, has_diffraction))


def detect_diffraction(frame: np.ndarray) -> bool:
    return False


def save_frame(buffer_name: str, q_buffer_ptr: int, worker_id: int):
    """Do this on the main thread I guess since it has meta information?"""
