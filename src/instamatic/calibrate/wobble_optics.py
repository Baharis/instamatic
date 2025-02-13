"""Set two optics settings and repeatedly oscillate between them."""

from __future__ import annotations

import argparse
import signal
import time
from textwrap import dedent
from typing import List

from instamatic import controller

running = True


def signal_handler(sig, frame):
    """Handles Ctrl+C gracefully."""
    global running
    print('\nCtrl+C detected. Stopping wobble...')
    running = False


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


class OpticsState:
    variables: List[str] = ['BeamShift']

    def __init__(self, ctrl) -> None:
        self.ctrl = ctrl
        self.values = {}

    @classmethod
    def current(cls, ctrl) -> OpticsState:
        self = cls(ctrl)
        self.get()
        return self

    def get(self) -> None:
        """Store current controller controller values in `self.values`"""
        for var in self.variables:
            self.values[var] = getattr(self.ctrl.tem, 'get' + var)()

    def set(self) -> None:
        """Apply stored `self.values` to current controller `self.ctrl."""
        for stored in self.variables:
            getattr(self.ctrl.tem, 'set' + stored)(*self.values.get(stored))

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}:\n'
        for stored, value in self.values.items():
            s += f'  - {stored:20} (Currently: {value})'
        return s


class OpticsWobbler:
    def __init__(self, ctrl) -> None:
        self.ctrl = ctrl
        self.o0 = OpticsState.current(ctrl)
        print('Simple script to "wobble" between two optics states.')
        print('The following beam variables will be wobbled:')
        print(self.o0)
        print()
        input(' >> Please set these variables for position 1 and press <ENTER>')
        self.o1 = OpticsState.current(ctrl)
        print(self.o1)
        print()
        input(' >> Please set these variables for position 2 and press <ENTER>')
        self.o2 = OpticsState.current(ctrl)
        print(self.o2)
        print()
        print('Oscillating continuously between positions 1 and 2...')
        print('In order to stop, press <CTRL+C> or <CTRL+BREAK>.')
        print('At termination, oscillated variables will reset to initial values.')
        self.states = [self.o0, self.o1, self.o2]


def wobble_optics(ctrl, **kwargs) -> None:
    ow = OpticsWobbler(ctrl)
    ctrl.cam.show_stream()
    try:
        while running:
            ow.o1.set()
            time.sleep(1.0)
            ow.o2.set()
            time.sleep(1.0)
    finally:
        print('Reverting oscillated variables to initial values:')
        print(ow.o0)
        ow.o0.set()


def main() -> None:
    description = dedent(__doc__)
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    options = parser.parse_args()

    ctrl = controller.initialize()
    kwargs = {}

    wobble_optics(ctrl, **kwargs)


if __name__ == '__main__':
    main()
