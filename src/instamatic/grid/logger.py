from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from instamatic.grid.geometry import GRID_REGISTRY, PeriodicConvexPolygonGridGeometry


class GridLogger:
    """Manages the real-time YAML representation of the grid in this format:

    window_type: square
    geometry:
        x: 11.111
        y: 22.222
        t: 0.033  # degrees
        w: 44.444
        s: 6.666
    intercepts:
        0:
          - [0.001, 0.002]
          - [5.003, 5.004]
          # ...
        -1: # fresh, not assigned to a window yet
          - [99.005, 99.006]
    """

    GRID_REGISTRY_INV = {v: k for k, v in GRID_REGISTRY.items()}

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = Path(Path.cwd() / 'grid.yaml' if path is None else path)

    def read(self) -> tuple[PeriodicConvexPolygonGridGeometry, dict]:
        """Read a grid object and intercepts from a grid.yaml log at path."""
        with open(self.path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        grid = GRID_REGISTRY[data['window_type']](**data['geometry'])
        return grid, data.get('intercepts', {})

    def write(self, grid: PeriodicConvexPolygonGridGeometry, intercepts: dict) -> None:
        """Dump the current geometry and intercepts to the YAML file."""
        window_type = self.GRID_REGISTRY_INV[grid.window_type]
        data = {
            'window_type': window_type,
            'geometry': grid.to_params(),
            'intercepts': {k: v.tolist() for k, v in intercepts.items()},
        }
        with open(self.path, 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)
