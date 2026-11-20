"""A CPU triangle renderer implemented with NumPy, without a 3D graphics engine."""

from .core import (
    Mesh,
    Renderer,
    look_at,
    perspective,
    clip_polygon,
    interpolate,
    coverage_mask,
)
