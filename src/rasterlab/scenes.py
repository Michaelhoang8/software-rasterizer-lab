"""Procedural source geometry and textures; no imported 3D renderer or assets."""

import numpy as np
from .core import Mesh


def torus(major=1.25, minor=0.46, around=64, tube=32):
    positions, normals, uv = [], [], []
    for i in range(around + 1):
        u = i / around * 2 * np.pi
        for j in range(tube + 1):
            v = j / tube * 2 * np.pi
            normal = np.array([np.cos(u) * np.cos(v), np.sin(v), np.sin(u) * np.cos(v)])
            positions.append(
                [
                    np.cos(u) * (major + minor * np.cos(v)),
                    minor * np.sin(v),
                    np.sin(u) * (major + minor * np.cos(v)),
                ]
            )
            normals.append(normal)
            uv.append([i / around * 4, j / tube * 2])
    triangles = []
    for i in range(around):
        for j in range(tube):
            a = i * (tube + 1) + j
            b = a + tube + 1
            triangles.extend([[a, b, a + 1], [a + 1, b, b + 1]])
    return Mesh(
        np.array(positions), np.array(normals), np.array(uv), np.array(triangles)
    )


def plane(size=4, y=-1):
    return Mesh(
        np.array(
            [[-size, y, -size], [size, y, -size], [size, y, size], [-size, y, size]]
        ),
        np.tile([0, 1, 0], (4, 1)),
        np.array([[0, 0], [4, 0], [4, 4], [0, 4]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )


def checker(size=256, checks=8, colors=((0.08, 0.7, 0.78), (0.67, 0.21, 0.83))):
    y, x = np.indices((size, size))
    mask = ((x * checks // size + y * checks // size) % 2).astype(bool)
    result = np.empty((size, size, 3))
    result[:] = colors[0]
    result[mask] = colors[1]
    # A narrow border makes the UV perspective difference easier to inspect.
    border = (x % (size // checks) < 2) | (y % (size // checks) < 2)
    result[border] *= 0.45
    return result


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    ry = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    a = 0.56
    c, s = np.cos(a), np.sin(a)
    rx = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    return ry @ rx
