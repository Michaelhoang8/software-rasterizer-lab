"""Independent ray/plane intersections validate pixel attributes, not just helpers."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rasterlab.core import Renderer, Mesh, perspective


class RayPlaneReference(unittest.TestCase):
    def test_pixel_uv_and_depth_match_geometric_intersections(self):
        points = np.array([[-0.8, -0.7, -2], [1, -0.6, -4], [-0.1, 0.9, -3]])
        uv = np.array([[0, 0], [2, 0], [0, 2]])
        mesh = Mesh(points, np.tile([0, 0, 1], (3, 1)), uv, np.array([[0, 1, 2]]))
        projection = perspective(np.pi / 3, 1, 1, 10)
        renderer = Renderer(60, 60)
        renderer.draw(
            mesh, np.eye(4), np.eye(4), projection, np.ones((2, 2, 3)), lit=False
        )
        pixels = np.argwhere(np.isfinite(renderer.depth))
        self.assertGreater(len(pixels), 50)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        basis = np.column_stack([points[1] - points[0], points[2] - points[0], normal])
        for y, x in pixels[:: max(1, len(pixels) // 25)]:
            ndc_x = (x + 0.5) / 60 * 2 - 1
            ndc_y = 1 - (y + 0.5) / 60 * 2
            ray = np.array([ndc_x / projection[0, 0], ndc_y / projection[1, 1], -1])
            distance = (normal @ points[0]) / (normal @ ray)
            hit = distance * ray
            a, b, off_plane = np.linalg.solve(basis, hit - points[0])
            self.assertAlmostEqual(off_plane, 0, places=12)
            expected_uv = np.array([1 - a - b, a, b]) @ uv
            np.testing.assert_allclose(
                renderer.uv[y, x], expected_uv, rtol=1e-12, atol=1e-12
            )
            homogeneous = projection @ np.append(hit, 1)
            self.assertAlmostEqual(
                renderer.depth[y, x],
                homogeneous[2] / homogeneous[3] * 0.5 + 0.5,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
