from __future__ import annotations

import inspect
import tkinter as tk
import tkinter.ttk as ttk
from functools import wraps
from typing import Callable, Protocol, Sequence, Union

import numpy as np


class GridWindowProtocol(Protocol):
    def __repr__(self) -> str: ...


class ProgressTable(ttk.Frame):
    """Use a ttk.TreeView to display the progress of scanning experiment."""

    COLUMNS = 'Geometry hits refls steps hits/step refls/step'.split()

    def __init__(self, parent: tk.Misc, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.tree = None
        self._build_tree()
        self._scan_geom: list[Union[int, str]] = []
        self._window_totals: tuple[int, int, int] = (0, 0, 0)  # hits, refls, steps

    def _build_tree(self) -> None:
        self.tree = ttk.Treeview(self, columns=self.COLUMNS, show='tree headings')

        for column in self.COLUMNS:
            self.tree.heading(column, text=column)

        self.tree.column('#0', width=30, stretch=True)
        self.tree.column('Geometry', anchor=tk.CENTER, width=60)
        self.tree.column('hits', anchor=tk.E, width=20)
        self.tree.column('refls', anchor=tk.E, width=20)
        self.tree.column('steps', anchor=tk.E, width=20)
        self.tree.column('hits/step', anchor=tk.E, width=20)
        self.tree.column('refls/step', anchor=tk.E, width=20)

        vsb = ttk.Scrollbar(orient='vertical', command=self.tree.yview)
        hsb = ttk.Scrollbar(orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(column=0, row=0, sticky='nsew', in_=self)
        vsb.grid(column=1, row=0, sticky='ns', in_=self)
        hsb.grid(column=0, row=1, sticky='ew', in_=self)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

    @staticmethod
    def _window_iid(window: int) -> str:
        return f'w:{window}'

    @staticmethod
    def _scan_iid(window: int, scan: int) -> str:
        return f'w:{window}/s:{scan}'

    @staticmethod
    def _step_iid(window: int, scan: int, step: int) -> str:
        return f'w:{window}/s:{scan}/p:{step}'

    def add_window(self, idx: int, window: GridWindowProtocol) -> None:
        """Add a new parent line to the tree called Window #."""
        window_iid = self._window_iid(idx)
        window_name = f'Window {idx:d}'
        geom = repr(window)
        values = (geom, '-', '-', '-', '-', '-')
        self.tree.insert('', tk.END, iid=window_iid, text=window_name, values=values)
        self._window_totals = (0, 0, 0)

    def add_scan(
        self,
        window: int,
        scan: int,
        x0: int,
        y0: int,
        direction: str,
        span: int,
        step: int,
    ):
        """Add a new child scan line to the tree called Scan #."""
        window_iid = self._window_iid(window)
        scan_iid = self._scan_iid(window, scan)
        scan_name = f'Scan {scan:d}'
        if direction.endswith('x'):
            geom = f'y: {y0}, x: {x0} -> {x0 + span}'
        else:
            geom = f'x: {x0}, y: {y0} -> {y0 + span}'
        values = (geom, '-', '-', -(-span // step), '-', '-')
        self.tree.insert(window_iid, tk.END, iid=scan_iid, text=scan_name, values=values)
        self._scan_geom = [x0, y0, direction, span]

    def fill_scan(
        self,
        window: int,
        scan: int,
        success: Union[np.ndarray, Sequence[Union[bool, None]]],
        n_peaks: Union[np.ndarray, Sequence[int]],
    ) -> None:
        """Add lines for successful experiments, update scan & column lines."""

        scan_iid = self._scan_iid(window, scan)
        window_iid = self._window_iid(window)
        x0, y0, direction, span = self._scan_geom
        step = span / len(success) * (-1 if direction.startswith('-') else 1)

        s_hits = sum(bool(s) for s in success)
        s_refls = sum(int(n) for ok, n in zip(success, n_peaks) if ok)
        s_steps = len(success)
        s_hits_per_step = s_hits / s_steps if s_steps else 0.0
        s_refls_per_step = s_refls / s_steps if s_steps else 0.0

        self.tree.set(scan_iid, 'hits', str(s_hits))
        self.tree.set(scan_iid, 'refls', str(s_refls))
        self.tree.set(scan_iid, 'steps', str(s_steps))
        self.tree.set(scan_iid, 'hits/step', f'{s_hits_per_step:.3g}')
        self.tree.set(scan_iid, 'refls/step', f'{s_refls_per_step:.3g}')

        w_hits = self._window_totals[0] + s_hits
        w_refls = self._window_totals[1] + s_refls
        w_steps = self._window_totals[2] + s_steps
        w_hits_per_step = w_hits / w_steps if w_steps else 0.0
        w_refls_per_step = w_refls / w_steps if w_steps else 0.0
        self._window_totals = (w_hits, w_refls, w_steps)

        self.tree.set(window_iid, 'hits', str(w_hits))
        self.tree.set(window_iid, 'refls', str(w_refls))
        self.tree.set(window_iid, 'steps', str(w_steps))
        self.tree.set(window_iid, 'hits/step', f'{w_hits_per_step:.3g}')
        self.tree.set(window_iid, 'refls/step', f'{w_refls_per_step:.3g}')

        for i, (ok, n) in enumerate(zip(success, n_peaks)):
            if not ok:
                continue
            step_name = f'Step {i:d}'
            step_iid = self._step_iid(window, scan, i)
            axis = direction[-1]
            geom = f'{axis}: {int((x0 if axis == "x" else y0) + i * step)}'
            values = (geom, '', int(n), '', '', '')
            self.tree.insert(scan_iid, tk.END, iid=step_iid, text=step_name, values=values)


def edits_progress(method: Callable) -> Callable:
    """Method decorator, captures calls to modify object's progress attr."""
    method_signature = inspect.signature(method)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        out = method(self, *args, **kwargs)
        if (progress := getattr(self, 'progress', None)) is not None:
            bound = method_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            kwargs = {k: v for k, v in bound.arguments.items() if k != 'self'}
            getattr(progress, method.__name__)(**kwargs)
        return out

    return wrapper


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Test progress listbox')
    listbox = ProgressTable(root)
    listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    listbox.add_window(0, 'Some geometry')
    listbox.add_scan(0, 0, 100, 200, '+x', 1000, 50)
    listbox.fill_scan(0, 0, success=[True, False, True, None, True], n_peaks=[12, 3, 8, 0, 21])
    listbox.add_scan(0, 1, 90, 210, '+x', 1020, 50)
    listbox.fill_scan(0, 1, success=[1, 0, 1, 0, 1, 1], n_peaks=[17, 3, 28, 0, 21, 19])

    root.mainloop()
