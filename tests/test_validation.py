"""Public entry points reject invalid geometry before allocating pixel work."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rasterlab.core import Renderer, Mesh, coverage_mask, perspective


class ValidationTests(unittest.TestCase):
    def test_integer_background_does_not_truncate_shaded_color(self):
        renderer = Renderer(4, 4, background=(0, 0, 0))
        vertices = np.array(
            [
                [-1, 1, 0, 1, 0, 0],
                [1, 1, 0, 1, 1, 0],
                [-1, -1, 0, 1, 0, 1],
            ]
        )
        renderer.triangle(vertices, np.array([[[0.2, 0.4, 0.6]]]))
        np.testing.assert_allclose(renderer.color[0, 0], [0.2, 0.4, 0.6])

    def test_invalid_dimensions_geometry_and_transforms(self):
        for size in [(0, 10), (10, -1), (3.5, 10), (True, 10)]:
            with self.assertRaises(ValueError):
                Renderer(*size)
        with self.assertRaises(ValueError):
            perspective(np.pi / 3, 1, 1, np.inf)
        with self.assertRaises(ValueError):
            coverage_mask([[0, 0], [1, np.nan], [2, 0]], 10, 10)
        renderer = Renderer(10, 10)
        with self.assertRaises(ValueError):
            renderer.triangle(np.full((3, 9), np.nan), np.ones((1, 1, 3)))
        mesh = Mesh(
            np.zeros((3, 3)),
            np.ones((3, 3)),
            np.zeros((3, 2)),
            np.array([[0, 1, 9]]),
        )
        with self.assertRaises(ValueError):
            renderer.draw(mesh, np.eye(4), np.eye(4), np.eye(4), np.ones((1, 1, 3)))


if __name__ == "__main__":
    unittest.main()
