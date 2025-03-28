from __future__ import annotations

from typing_extensions import Annotated

int_nm = Annotated[int, 'Length expressed in nanometers']
float_deg = Annotated[float, 'Angle expressed in degrees']
