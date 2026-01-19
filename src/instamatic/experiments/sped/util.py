from __future__ import annotations

from typing import NamedTuple, Optional

from fontTools.misc.cython import returns


class SPEDLoc(NamedTuple):
    """Universal SPED indexing/locating format for grid, scans, frames etc."""

    grid_i: int
    grid_j: int
    scan: Optional[int] = None
    step: Optional[int] = None

    @property
    def name(self) -> str:
        return 'SPED_' + '_'.join(str(i) for i in self if i is not None)
