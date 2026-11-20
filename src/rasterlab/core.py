"""Homogeneous clipping, top-left rasterization, depth and attribute interpolation.

Convention: column vectors; right-handed view looking down -Z; OpenGL clip volume
-w <= x,y,z <= w; screen origin top-left; samples at pixel centers. Calculations
use float64. Color values are treated as display-space RGB for this educational
renderer: lighting is intentionally approximate, not physically based.
"""

from dataclasses import dataclass
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError("Vector must be finite")
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(length == 0):
        raise ValueError("Cannot normalize a zero vector")
    return v / length


def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, dtype=np.float64)
    forward = unit(np.asarray(target, dtype=np.float64) - eye)
    right = unit(np.cross(forward, up))
    camera_up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, camera_up, -forward])
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def perspective(fov_y, aspect, near, far):
    if (
        not np.isfinite([fov_y, aspect, near, far]).all()
        or not 0 < fov_y < np.pi
        or aspect <= 0
        or not 0 < near < far
    ):
        raise ValueError("Invalid perspective parameters")
    f = 1 / np.tan(fov_y / 2)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float64,
    )


PLANES = np.array(
    [
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, -1, 1],
    ],
    dtype=np.float64,
)


def clip_polygon(vertices):
    """Sutherland-Hodgman clipping; first four columns are homogeneous positions.

    Remaining attributes are interpolated at the same edge parameter, before
    perspective division. An empty (0,k) array represents an invisible polygon.
    """
    polygon = np.asarray(vertices, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] < 4 or not np.isfinite(polygon).all():
        raise ValueError("Expected finite vertices with homogeneous position")
    columns = polygon.shape[1]
    for plane in PLANES:
        if not len(polygon):
            break
        output = []
        previous = polygon[-1]
        previous_distance = previous[:4] @ plane
        for current in polygon:
            distance = current[:4] @ plane
            inside, previous_inside = distance >= 0, previous_distance >= 0
            if inside != previous_inside:
                t = previous_distance / (previous_distance - distance)
                output.append(previous + t * (current - previous))
            if inside:
                output.append(current)
            previous, previous_distance = current, distance
        polygon = np.array(output, dtype=np.float64).reshape(-1, columns)
    return polygon


def _edge(a, b, x, y):
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def _top_left(a, b):
    dx, dy = b - a
    return dy < 0 or dy == 0 and dx > 0


def _coverage(screen, width, height):
    """Return bbox, barycentrics and mask for positive-orientation screen vertices."""
    area = _edge(screen[0], screen[1], screen[2, 0], screen[2, 1])
    if area <= 0:
        return None
    low = np.maximum(np.ceil(screen.min(axis=0) - 0.5).astype(int), [0, 0])
    high = np.minimum(
        np.floor(screen.max(axis=0) - 0.5).astype(int), [width - 1, height - 1]
    )
    if np.any(high < low):
        return None
    x = np.arange(low[0], high[0] + 1)[None, :] + 0.5
    y = np.arange(low[1], high[1] + 1)[:, None] + 0.5
    edges = np.stack(
        np.broadcast_arrays(
            *[_edge(screen[a], screen[b], x, y) for a, b in [(1, 2), (2, 0), (0, 1)]]
        ),
        axis=-1,
    )
    inclusive = np.array(
        [_top_left(screen[a], screen[b]) for a, b in [(1, 2), (2, 0), (0, 1)]]
    )
    mask = np.all((edges > 0) | ((edges == 0) & inclusive), axis=-1)
    return low, high, edges / area, mask


def coverage_mask(screen, width, height):
    screen = np.asarray(screen, dtype=np.float64).copy()
    if screen.shape != (3, 2) or not np.isfinite(screen).all():
        raise ValueError("Coverage requires three finite screen vertices")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive integers")
    if _edge(screen[0], screen[1], screen[2, 0], screen[2, 1]) < 0:
        screen[[1, 2]] = screen[[2, 1]]
    output = np.zeros((height, width), dtype=bool)
    covered = _coverage(screen, width, height)
    if covered:
        low, high, _, mask = covered
        output[low[1] : high[1] + 1, low[0] : high[0] + 1] = mask
    return output


def interpolate(attributes, barycentrics, reciprocal_w, mode="perspective"):
    attributes = np.asarray(attributes, dtype=np.float64)
    barycentrics = np.asarray(barycentrics, dtype=np.float64)
    if mode == "perspective":
        weights = barycentrics * np.asarray(reciprocal_w)
        weights = weights / weights.sum(axis=-1, keepdims=True)
    elif mode == "affine":
        weights = barycentrics
    else:
        raise ValueError("Interpolation must be perspective or affine")
    return weights @ attributes


@dataclass(frozen=True)
class Mesh:
    positions: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray


class Renderer:
    def __init__(self, width, height, background=(0.055, 0.045, 0.1)):
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
        ):
            raise ValueError("Frame dimensions must be positive integers")
        if np.asarray(background).shape != (3,) or not np.isfinite(background).all():
            raise ValueError("Background must have three finite color components")
        self.width, self.height = width, height
        self.color = np.broadcast_to(
            np.asarray(background, dtype=np.float64), (height, width, 3)
        ).copy()
        self.depth = np.full((height, width), np.inf)
        self.uv = np.full((height, width, 2), np.nan)
        self.coverage = np.zeros((height, width), dtype=np.int32)
        self.stats = {"input_triangles": 0, "clipped_triangles": 0, "shaded_samples": 0}

    def draw(
        self, mesh, model, view, projection, texture, mode="perspective", lit=True
    ):
        if mode not in ("perspective", "affine"):
            raise ValueError("Interpolation must be perspective or affine")
        for matrix in (model, view, projection):
            if np.asarray(matrix).shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("Transforms must be finite 4x4 matrices")
        count = len(mesh.positions)
        if (
            mesh.positions.shape != (count, 3)
            or mesh.normals.shape != (count, 3)
            or mesh.uv.shape != (count, 2)
        ):
            raise ValueError("Mesh position, normal and UV shapes do not agree")
        if not all(
            np.isfinite(a).all() for a in (mesh.positions, mesh.normals, mesh.uv)
        ):
            raise ValueError("Mesh attributes must be finite")
        if (
            mesh.triangles.ndim != 2
            or mesh.triangles.shape[1] != 3
            or not np.issubdtype(mesh.triangles.dtype, np.integer)
        ):
            raise ValueError("Triangle indices must be an Nx3 integer array")
        if np.any(mesh.triangles < 0) or np.any(mesh.triangles >= count):
            raise ValueError("Triangle index is outside the vertex array")
        texture = self._texture(texture)
        homogeneous = np.column_stack([mesh.positions, np.ones(len(mesh.positions))])
        clip = homogeneous @ (projection @ view @ model).T
        # Correct for nonuniform model scaling; lighting uses world-space normals.
        normals = mesh.normals @ np.linalg.inv(model[:3, :3])
        vertices = np.column_stack([clip, mesh.uv, normals])
        for indices in mesh.triangles:
            self.stats["input_triangles"] += 1
            polygon = clip_polygon(vertices[indices])
            for i in range(1, len(polygon) - 1):
                self.stats["clipped_triangles"] += 1
                self._triangle(polygon[[0, i, i + 1]], texture, mode, lit)

    def triangle(self, vertices, texture, mode="perspective", lit=False):
        """Rasterize one already-clipped triangle with clip/UV/normal columns."""
        vertices = np.asarray(vertices, dtype=np.float64)
        allowed_shapes = ((3, 9),) if lit else ((3, 6), (3, 9))
        if vertices.shape not in allowed_shapes or not np.isfinite(vertices).all():
            raise ValueError(
                "Expected three finite clip/UV vertices, plus normals when lit"
            )
        if mode not in ("perspective", "affine"):
            raise ValueError("Interpolation must be perspective or affine")
        self._triangle(vertices, self._texture(texture), mode, lit)

    @staticmethod
    def _texture(texture):
        texture = np.asarray(texture, dtype=np.float64)
        if (
            texture.ndim != 3
            or texture.shape[2] != 3
            or min(texture.shape[:2]) == 0
            or not np.isfinite(texture).all()
        ):
            raise ValueError("Texture must be a nonempty finite HxWx3 array")
        return texture

    def _triangle(self, vertices, texture, mode, lit):
        vertices = np.asarray(vertices, dtype=np.float64).copy()
        if np.any(vertices[:, 3] <= 0):
            return  # Only a degenerate homogeneous origin can survive clipping here.
        ndc = vertices[:, :3] / vertices[:, 3, None]
        screen = np.column_stack(
            [(ndc[:, 0] + 1) * self.width / 2, (1 - ndc[:, 1]) * self.height / 2]
        )
        if _edge(screen[0], screen[1], screen[2, 0], screen[2, 1]) < 0:
            vertices[[1, 2]] = vertices[[2, 1]]
            ndc[[1, 2]] = ndc[[2, 1]]
            screen[[1, 2]] = screen[[2, 1]]
        covered = _coverage(screen, self.width, self.height)
        if covered is None:
            return
        low, high, bary, inside = covered
        region = np.s_[low[1] : high[1] + 1, low[0] : high[0] + 1]
        self.coverage[region] += inside.astype(np.int32)
        # Window depth is affine in screen barycentrics. Applying 1/w again is wrong.
        depth = (bary @ ndc[:, 2]) * 0.5 + 0.5
        valid = inside & (depth >= 0) & (depth <= 1) & (depth < self.depth[region])
        if not np.any(valid):
            return
        attrs = interpolate(vertices[:, 4:], bary[valid], 1 / vertices[:, 3], mode)
        uv = attrs[:, :2]
        tex = np.asarray(texture, dtype=np.float64)
        tx = np.floor(np.mod(uv[:, 0], 1) * tex.shape[1]).astype(int) % tex.shape[1]
        ty = np.floor(np.mod(uv[:, 1], 1) * tex.shape[0]).astype(int) % tex.shape[0]
        color = tex[ty, tx, :3]
        if lit:
            normal = attrs[:, 2:]
            length = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal / np.maximum(length, 1e-15)
            light = unit([-0.4, 0.85, 0.65])
            diffuse = np.maximum(normal @ light, 0)
            rim = (1 - np.abs(normal[:, 2])) ** 3
            color = color * (0.22 + 0.78 * diffuse[:, None]) + rim[:, None] * np.array(
                [0.12, 0.035, 0.18]
            )
        self.color[region][valid] = np.clip(color, 0, 1)
        self.depth[region][valid] = depth[valid]
        self.uv[region][valid] = uv
        self.stats["shaded_samples"] += int(valid.sum())

    def image(self):
        return np.round(np.clip(self.color, 0, 1) * 255).astype(np.uint8)
