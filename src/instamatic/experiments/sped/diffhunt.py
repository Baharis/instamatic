from __future__ import annotations

import multiprocessing as mp
import multiprocessing.shared_memory
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import Literal

from instamatic._typing import AnyPath

N_PROCESSORS = 4


mp.set_start_method('spawn', force=True)

Task = Literal['PROCESS', 'WRITE', 'TERMINATE']
Event = Literal['PROCESSING', 'PROCESSED', 'SWITCHED', 'TERMINATED']


@dataclass(frozen=True)
class Command:
    """Schema used to communicate commands from dispatcher to any worker."""

    task: Task
    buffer_name: Optional[str] = None
    buffer_pointer: Optional[int] = None
    kwargs: Optional[dict] = None


@dataclass(frozen=True)
class Feedback:
    """Schema used to communicate feedback from any worker to dispatcher."""

    event: Event
    worker_id: int
    buffer_name: Optional[str] = None
    buffer_pointer: Optional[int] = None
    details: Optional[dict] = None


class DiffHuntDispatcher:
    """Proxy class: ask workers on other processes if image has diffraction"""

    @dataclass
    class Worker:
        buffer: str = ''
        busy: bool = False
        pointer: Optional[int] = None
        process: mp.Process = None

    @dataclass
    class Buffer:
        frames: np.ndarray
        name: str = field(default_factory=lambda: uuid.uuid4().hex)
        pointer: int = 0
        pointers: set[int] = field(default_factory=set)  # currently processed
        workers: set[int] = field(default_factory=set)  # attached to buffer

    def __init__(self, shape, dtype):
        self.shape: tuple[int, int] = shape
        self.dtype: np.dtype = dtype

        self.commands: mp.Queue[Command] = mp.Queue()
        self.feedback: mp.Queue[Feedback] = mp.Queue()
        self.workers: dict[int, DiffHuntDispatcher.Worker] = self.initialize_workers()
        self.buffers: dict[str, DiffHuntDispatcher.Buffer] = {}
        self.history = pd.DataFrame(columns=['buffer', 'pointer', 'has_diffraction', 'header'])
        self.history.set_index(['buffer', 'pointer'], inplace=True)

    def initialize_workers(self) -> dict[int, DiffHuntDispatcher.Worker]:
        """Run once at the start of experiment to spawn eval processes."""
        workers = {}
        for i in range(N_PROCESSORS):
            worker = DiffHuntWorker(i, self.commands, self.feedback, self.dtype)
            worker.start()
            workers[i] = self.Worker(process=worker)
        return workers

    def switch_buffer(self, n_frames: int = 100, name: str = None) -> None:
        """Configure a new mp shared memory space to buffer a frame stack."""
        shape = (n_frames, self.shape[0], self.shape[1])
        size = np.prod(shape) * np.dtype(self.dtype).itemsize
        shm = mp.shared_memory.SharedMemory(name=name, create=True, size=size)
        frames = np.ndarray(shape, dtype=self.dtype, buffer=shm.buf)
        self.buffers[name] = self.Buffer(frames=frames, name=name)

    def write_buffer(self, path: AnyPath) -> None:
        """Save all the frames with diffraction in an active buffer."""
        ab = list(self.buffers.values())[-1]  # last i.e. active buffer
        h = self.history
        to_save = h[(h['buffer'] == ab.name) & h['has_diffraction']]
        for ptr, h in to_save[['pointer', 'header']]:
            self.emit('WRITE', ab.name, ptr, kwargs={'path': path, 'header': h})

    # COMMANDING METHODS THAT DISPATCH COMMANDS TO WORKERS

    def emit(self, task: Task, *args, **kwargs) -> None:
        """Shorthand to create and put Command in the self.commands queue."""
        self.commands.put(Command(task, *args, **kwargs))

    def process(self, frame: np.ndarray, header: Optional[dict]) -> None:
        """Request 'PROCESS_FRAME' from buffer stored on some shared memory."""
        ab = list(self.buffers.values())[-1]  # last i.e. active buffer
        if ab.pointer >= ab.frames.shape[0]:
            raise RuntimeError(f'{ab.name} buffer overflow')
        ab.frames[ab.pointer, :, :] = frame
        self.emit('PROCESS', buffer_name=ab.name, buffer_pointer=ab.pointer)
        ab.pointers.add(ab.pointer)
        ab.pointer += 1
        self.history.loc[(ab.name, ab.pointer), 'header'] = header
        self.history.loc[(ab.name, ab.pointer), 'has_diffraction'] = None

    def terminate_workers(self) -> None:
        """Command all workers to 'TERMINATE' and report the success."""
        for _ in self.workers:
            self.emit('TERMINATE')

    # HANDLE FEEDBACK INCOMING FROM THE WORKERS

    def handle_feedback(self, stop_event: threading.Event) -> None:
        """To be called in a separate thread to handle incoming feedback."""
        while not stop_event.is_set():
            try:
                fb: Feedback = self.feedback.get(timeout=0.05)
            except queue.Empty:
                continue

            worker = self.workers.get(fb.worker_id)

            if fb.event == 'PROCESSING':
                worker.busy = True
                worker.buffer = fb.buffer_name
                worker.pointer = fb.buffer_pointer
                buffer = self.buffers.get(fb.buffer_name)
                buffer.workers.add(fb.worker_id)
                buffer.pointers.add(fb.buffer_pointer)

            elif fb.event == 'PROCESSED_FRAME':
                idx = (fb.buffer_name, fb.buffer_pointer)
                has = fb.details.get('has_diffraction', 'False')
                self.history.at[idx, 'has_diffraction'] = has
                worker.busy = False
                worker.pointer = None
                buffer = self.buffers.get(fb.buffer_name)
                buffer.pointers.discard(fb.buffer_pointer)

            elif fb.event == 'SWITCHED':
                if old_buffer := worker.buffer:
                    self.buffers.get(old_buffer).workers.discard(fb.worker_id)
                worker.buffer = fb.buffer_name
                self.buffers.get(fb.buffer_name).workers.add(fb.worker_id)
                self._maybe_release_buffer(self.buffers.get(fb.buffer_name))

            elif fb.event == 'TERMINATE':
                self._terminate_worker(fb.worker_id)

            else:
                raise ValueError(f'Unknown feedback event {fb.event}')

    def _maybe_release_buffer(self, buffer: DiffHuntDispatcher.Buffer):
        """If the buffer has no workers and no plans, release its memory."""
        if not buffer.workers and not buffer.pointers:
            del self.buffers[buffer.name]
            try:
                buffer.frames.base.close()
                buffer.frames.base.unlink()
            except Exception as e:
                print(f'Warning: could not release buffer {buffer.name}: {e}')

    def _terminate_worker(self, worker_id: int) -> None:
        """Once the worker is ready to terminate, join and close it."""
        worker = self.workers.pop(worker_id)
        if worker.buffer:
            buffer = self.buffers.get(worker.buffer)
            if buffer:
                buffer.workers.discard(worker_id)
                self._maybe_release_buffer(buffer)
        worker.process.join()
        worker.process.close()


class DiffHuntWorker(mp.Process):
    """Stateful diffraction-hunting work process handled by the dispatcher."""

    def __init__(
        self,
        worker_id: int,
        commands: mp.Queue,
        feedback: mp.Queue,
        dtype: np.dtype,
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.commands = commands
        self.feedback = feedback
        self.dtype = dtype

        self.buffer: np.ndarray = np.array([], dtype=dtype)
        self.buffer_name: str = ''

    def emit(self, event: Event, *args, **kwargs) -> None:
        """Put worker_id followed by all args in the self.feedback queue."""
        self.feedback.put(Feedback(event, self.worker_id, *args, **kwargs))

    def run(self) -> None:
        """Main loop passing incoming commands to respective methods."""
        while True:
            cmd, *args = self.commands.get(block=True)

            if cmd == 'TERMINATE':
                self.emit('TERMINATED')
                return

            if cmd == 'PROCESS_FRAME':
                self._process_frame(*args)
                continue

            raise ValueError(f'Unknown command: {cmd}')

    def _process_frame(
        self,
        buffer_name: str,
        buffer_shape: tuple,
        frame_index: int,
    ) -> None:
        """Handles the 'PROCESS FRAME' command."""

        if buffer_name != self.buffer_name:
            self.buffer_name = buffer_name
            shm = mp.shared_memory.SharedMemory(name=buffer_name)
            self.buffer = np.ndarray(buffer_shape, dtype=self.dtype, buffer=shm.buf)
            self.emit('SWITCHED', buffer_name=buffer_name)

        self.emit('PROCESSING', buffer_name=buffer_name, buffer_pointer=frame_index)
        frame = self.buffer[frame_index]
        d = {'has_diffraction': detect_diffraction(frame)}
        self.emit('PROCESSED', buffer_name=buffer_name, buffer_pointer=frame_index, details=d)


def detect_diffraction(frame: np.ndarray) -> bool:
    return False


def save_frame(buffer_name: str, q_buffer_ptr: int, worker_id: int):
    """Do this on the main thread I guess since it has meta information?"""
