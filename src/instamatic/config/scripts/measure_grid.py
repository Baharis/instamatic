from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from typing_extensions import Literal, Self

from instamatic._typing import float_nm, int_nm
from instamatic.utils.iterating import pairwise

if TYPE_CHECKING:
    from instamatic.controller import TEMController


global _ctrl
_ctrl: TEMController


X = np.array([1, 0], dtype=float)
Y = np.array([0, 1], dtype=float)
Array = np.ndarray
Vector2 = Sequence[float]


class Sweeper:
    """A simple descriptor of stage movement with fixed heading."""

    step: int_nm = 1000

    def __init__(self, origin: Vector2, heading: Vector2) -> None:
        self.origin = np.array([origin[0], origin[1]], dtype=float)
        self.heading = np.array([heading[0], heading[1]], dtype=float)
        self.position = np.array([origin[0], origin[1]], dtype=float)


class EdgeSweeper(Sweeper):
    """Used to determine the edge of the stage based on camera feedback."""

    step: int_nm = 10_000  # largest step size allowed
    precision: int_nm = 1  # smallest step size allowed
    threshold: float = 0.01  # fraction of light_max that signals the edge
    light_max: int = -1  # maximum light observed at any point by any sweeper

    def peak(self) -> int:
        """Return light (image sum) at current position, update light max."""
        light = int(_ctrl.get_image().sum())
        EdgeSweeper.light_max = max(light, self.light_max)
        return light

    def walk(self, dx: float, dy: float) -> None:
        """Change sweeper position by a (dx, dy) vector."""
        x, y = _ctrl.stage.xy
        x = int(x + dx)
        y = int(y + dy)
        _ctrl.stage.set(x=x, y=y)
        self.position = np.array([x, y], dtype=float)


class CrudeEdgeSweeper(EdgeSweeper):
    """Moves monotonously, stops when intensity fract < max * threshold."""

    def sweep(self) -> None:
        """Walk steps into heading until peaked light is below threshold."""
        _ctrl.stage.set(int(self.origin[0]), int(self.origin[1]))
        light_here: int = self.peak()
        while light_here > self.light_max * self.threshold:
            dx: float = self.heading[0].item() * self.step
            dy: float = self.heading[1].item() * self.step
            self.walk(dx=dx, dy=dy)
            light_here: int = self.peak()


class BinaryEdgeSweeper(EdgeSweeper):
    """A stage-state descriptor used to binary-search the grid edge."""

    def breed(self, other: Self) -> Self:
        """Return a new instance with mean heading and position."""
        o = (self.position + other.position) / 2
        h = (s := self.heading + other.heading) / float(np.linalg.norm(s))
        return self.__class__(origin=o, heading=h)

    def sweep(self) -> None:
        """Bin-search the edge based on peaked light vs max * threshold."""
        _ctrl.stage.set(int(self.origin[0]), int(self.origin[1]))
        _step = self.step
        _mult = 1.0
        while not self.precision > _step > -self.precision:
            dx: float = self.heading[0].item() * _step
            dy: float = self.heading[1].item() * _step
            self.walk(dx=dx, dy=dy)
            light_here = self.peak()
            if light_here > self.threshold * self.light_max:
                _step = _mult * abs(_step)
            else:
                _mult = 0.5
                _step = -_mult * abs(_step)


class RectangularGridWindow:
    """Describes one rectangular window without assumptions about the grid.

    Geometry is described using five immutable float parameters (nm / radian):

    - center_x: coordinate of the window center on the X axis;
    - center_y: coordinate of the window center on the Y axis;
    - width: length of window side aligned with the direction of X axis;
    - height: length of window side aligned with the direction or Y axis;
    - theta: signed angle from X towards X-aligned edge (positive towards Y);
    """

    def __init__(self, x: float, y: float, w: float, h: float, t: float):
        t = (t + (np.pi / 2)) % np.pi - (np.pi / 2)  # cast to [-pi/2, pi/2]
        if not -np.pi / 4 < t < np.pi / 4:  # cast to [-pi/4, pi/4]
            w, h, t = h, w, (np.pi - t) % np.pi - np.pi / 2

        self.center_x: float_nm = x
        self.center_y: float_nm = y
        self.width = w = abs(w)
        self.height = h = abs(h)
        self.theta: float = t  # expressed in radian

        self.center = c = np.array([x, y], dtype=float)
        self.w_axis = wa = 0.5 * w * np.array([np.cos(t), np.sin(t)], dtype=float)
        self.h_axis = ha = 0.5 * h * np.array([-np.sin(t), np.cos(t)], dtype=float)
        self.corners = np.vstack([c + wa + ha, c + wa - ha, c - wa - ha, c - wa + ha])

    @staticmethod
    def edge_dist2_sum(geom: tuple[float, float, float, float, float], xys: Array) -> float:
        """scipy.optimize.minimize fitting func; for geometry see cls docs."""
        center_x, center_y, width, height, theta = geom
        center = np.array([center_x, center_y], dtype=float)
        w_axis = 0.5 * width * np.array([np.cos(theta), np.sin(theta)])
        h_axis = 0.5 * height * np.array([-np.sin(theta), np.cos(theta)])
        w_axis_n = w_axis / np.linalg.norm(w_axis)
        h_axis_n = h_axis / np.linalg.norm(h_axis)
        d1 = np.abs(np.dot(xys - (center + w_axis), w_axis_n))
        d2 = np.abs(np.dot(xys - (center - w_axis), w_axis_n))
        d3 = np.abs(np.dot(xys - (center + h_axis), h_axis_n))
        d4 = np.abs(np.dot(xys - (center - h_axis), h_axis_n))
        return np.sum(np.min([d1, d2, d3, d4], axis=0) ** 2)

    @classmethod
    def from_star_search(cls, order: Literal[1, 2, 3, 4, 5] = 3) -> Self:
        """Return new using `EdgeSweeper`s scanning around current position."""
        origin = np.array(*_ctrl.stage.xy, dtype=float)
        css: dict[str, CrudeEdgeSweeper] = {
            '+X': CrudeEdgeSweeper(origin=origin, heading=+X),
            '-X': CrudeEdgeSweeper(origin=origin, heading=-X),
            '+Y': CrudeEdgeSweeper(origin=origin, heading=+Y),
            '-Y': CrudeEdgeSweeper(origin=origin, heading=-Y),
        }
        for cs in css.values():
            cs.sweep()
        center_x = (css['+X'].position[0] - css['-X'].position[0]) / 2
        center_y = (css['+Y'].position[0] - css['-Y'].position[0]) / 2
        center = np.array([center_x, center_y], dtype=float)

        bss = [BinaryEdgeSweeper(origin=center, heading=d) for d in (X, Y, -X, -Y)]
        for bs in bss:
            bs.sweep()

        def bisectors(sweepers: list[BinaryEdgeSweeper]) -> list[BinaryEdgeSweeper]:
            new = [a.breed(b) for a, b in pairwise([*sweepers, sweepers[0]])]
            for ns in new:
                ns.sweep()
            return new

        for _ in range(1, order):
            bss = list(chain.from_iterable(zip(bss, bisectors(bss))))

        edge_xy = np.vstack([bs.position for bs in bss])  # Nx2
        return cls.from_edge_xys(edge_xy)

    @classmethod
    def from_edge_xys(cls, edge_xys: Array) -> Self:
        """Return new by fitting the edge to a Nx2 list of edge positions."""
        xys_com = np.mean(edge_xys, axis=0)
        xys_deltas = edge_xys - xys_com
        xys_cov = np.cov(xys_deltas.T)
        eigenvalues, eigenvectors = np.linalg.eigh(xys_cov)
        eigenvector_proj = xys_deltas @ eigenvectors
        width0 = eigenvector_proj[:, 1].max() - eigenvector_proj[:, 1].min()
        height0 = eigenvector_proj[:, 0].max() - eigenvector_proj[:, 0].min()
        theta0 = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
        guess = np.array([xys_com[0], xys_com[1], width0, height0, theta0])
        res = minimize(cls.edge_dist2_sum, guess, args=(edge_xys,), method='Powell')
        return cls(*res.x)

    def x_intersections(self, y: int_nm) -> Optional[tuple[float, float]]:
        """Return (x_min, x_max) for a horizontal line intersecting at y."""
        intersection_xs: list[float] = []

        for x1, y1, x2, y2 in pairwise([*self.corners, self.corners[0]]):
            if y1 == y2:  # work with edge case , close to zero
                continue
            intersection_fraction = (y - y1) / (y2 - y1)
            if not 0 < intersection_fraction < 1:
                continue  # does not intersect
            intersection_xs.append(x1 + (x2 - x1) * intersection_fraction)

        if len(intersection_xs) < 2:
            return None
        return min(intersection_xs), max(intersection_xs)

    def plan_x_sweeping(self, step: int_nm = 1000) -> Iterator[XSweep]:
        """Yield `XSweep`s every `step` nm across the whole grid window."""
        ys = [c[1] for c in self.corners]
        y_min, y_max = int(min(ys)), int(max(ys))

        for i, y in enumerate(range(y_min, y_max + 1, step)):
            if x_intersections := self.x_intersections(y):
                x_start = int(x_intersections[i % 2])
                x_end = int(x_intersections[(i + 1) % 2])
                yield XSweep(i=i, x=x_start, y=y, d=x_end - x_start)


@dataclass
class XSweep:
    i: int  # sweep index or identifier
    x: int_nm  # starting x position
    y: int_nm  # starting y position
    d: int_nm  # total x span to cover
