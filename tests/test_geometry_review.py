"""Independent analytic anchors for clipping, coverage and interpolation."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rasterlab.core import (
    clip_polygon,
    coverage_mask,
    interpolate,
    look_at,
    perspective,
    Renderer,
)


class GeometryReview(unittest.TestCase):
    def test_each_homogeneous_clip_plane_preserves_edge_attributes(self):
        for axis in range(3):
            for sign in [-1, 1]:
                with self.subTest(axis=axis, sign=sign):
                    vertices = np.zeros((3, 6))
                    vertices[:, 3] = 1
                    vertices[0, axis] = sign * 2
                    vertices[1, (axis + 1) % 3] = 0.6
                    vertices[2, (axis + 1) % 3] = -0.6
                    vertices[:, 4:] = [[10, 4], [20, 8], [30, 12]]
                    clipped = clip_polygon(vertices)
                    self.assertEqual(len(clipped), 4)
                    self.assertTrue(
                        (np.abs(clipped[:, :3]) <= clipped[:, 3, None] + 1e-14).all()
                    )
                    boundary = clipped[np.isclose(clipped[:, axis], sign)]
                    self.assertEqual(len(boundary), 2)
                    np.testing.assert_allclose(sorted(boundary[:, 4]), [15, 20])
                    np.testing.assert_allclose(sorted(boundary[:, 5]), [6, 8])

    def test_homogeneous_edge_parameter_is_not_postdivide_parameter(self):
        vertices = np.array(
            [[4, 0, 0, 2, 10], [0, 0.6, 0, 1, 40], [0, -0.6, 0, 1, 70]], float
        )
        clipped = clip_polygon(vertices)
        boundary = clipped[np.isclose(clipped[:, 0], clipped[:, 3])]
        self.assertEqual(len(boundary), 2)
        np.testing.assert_allclose(boundary[:, 0], 4 / 3)
        np.testing.assert_allclose(sorted(boundary[:, 4]), [30, 50])
        entirely_outside = np.array(
            [[2, 0, 0, 1], [3, 0.2, 0, 1], [4, -0.2, 0, 1]], float
        )
        self.assertEqual(clip_polygon(entirely_outside).shape, (0, 4))

    def test_top_left_diagonal_has_neither_cracks_nor_double_coverage(self):
        quad = np.array([[1, 1], [7, 1], [7, 7], [1, 7]], float)
        expected = np.zeros((9, 9), int)
        expected[1:7, 1:7] = 1
        for indices in [
            ([0, 1, 2], [0, 2, 3]),
            ([2, 1, 0], [3, 2, 0]),
            ([0, 1, 3], [1, 2, 3]),
        ]:
            total = sum(
                coverage_mask(quad[list(triangle)], 9, 9).astype(int)
                for triangle in indices
            )
            np.testing.assert_array_equal(total, expected)
        degenerate = np.array([[1, 1], [2, 2], [3, 3]], float)
        self.assertFalse(coverage_mask(degenerate, 9, 9).any())

    def test_offscreen_coverage_is_clamped_without_wraparound(self):
        quad = np.array([[-2, -2], [3, -2], [3, 3], [-2, 3]], float)
        combined = coverage_mask(quad[[0, 1, 2]], 5, 5).astype(int) + coverage_mask(
            quad[[0, 2, 3]], 5, 5
        ).astype(int)
        expected = np.zeros((5, 5), int)
        expected[:3, :3] = 1
        np.testing.assert_array_equal(combined, expected)

    def test_uv_anchor_uses_reciprocal_w_and_depth_does_not(self):
        # Screen triangle is (0,0), (8,0), (0,8). Pixel [1,1] samples
        # (1.5,1.5), whose barycentrics are (5/8,3/16,3/16).
        ndc = np.array([[-1, 1, -0.6], [1, 1, 0.2], [-1, -1, 0.8]])
        w = np.array([1.0, 2.0, 4.0])
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        vertices = np.column_stack([ndc * w[:, None], w, uv])
        bary = np.array([5 / 8, 3 / 16, 3 / 16])
        expected_uv = np.array([6 / 49, 3 / 49])
        np.testing.assert_allclose(
            interpolate(uv, bary, 1 / w), expected_uv, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            interpolate(uv, bary, 1 / w, mode="affine"), [3 / 16, 3 / 16]
        )
        renderer = Renderer(8, 8)
        renderer.triangle(vertices, np.ones((2, 2, 3)))
        np.testing.assert_allclose(renderer.uv[1, 1], expected_uv, atol=1e-15)
        expected_depth = (bary @ ndc[:, 2]) * 0.5 + 0.5
        self.assertAlmostEqual(renderer.depth[1, 1], expected_depth, places=14)
        wrong_depth = float(interpolate(ndc[:, 2, None], bary, 1 / w)[0]) * 0.5 + 0.5
        self.assertGreater(abs(renderer.depth[1, 1] - wrong_depth), 0.05)

    def test_depth_test_retains_nearest_triangle_independent_of_submission_order(self):
        xy = np.array([[-1, 1], [1, 1], [-1, -1]], float)

        def triangle(z):
            return np.column_stack([xy, np.full(3, z), np.ones(3), np.zeros((3, 2))])

        red = np.array([[[1.0, 0.0, 0.0]]])
        blue = np.array([[[0.0, 0.0, 1.0]]])
        for sequence in [((-0.5, red), (0.5, blue)), ((0.5, blue), (-0.5, red))]:
            renderer = Renderer(8, 8)
            for depth, texture in sequence:
                renderer.triangle(triangle(depth), texture)
            np.testing.assert_array_equal(renderer.color[1, 1], [1, 0, 0])
            self.assertAlmostEqual(renderer.depth[1, 1], 0.25)
            self.assertEqual(renderer.coverage[1, 1], 2)

    def test_camera_basis_and_near_far_mapping(self):
        eye = np.array([3.0, 2.0, 5.0])
        target = np.array([-1.0, 0.5, -2.0])
        view = look_at(eye, target)
        np.testing.assert_allclose(view @ np.r_[eye, 1], [0, 0, 0, 1], atol=1e-14)
        target_camera = view @ np.r_[target, 1]
        np.testing.assert_allclose(target_camera[:2], [0, 0], atol=1e-14)
        self.assertAlmostEqual(target_camera[2], -np.linalg.norm(target - eye))
        np.testing.assert_allclose(view[:3, :3] @ view[:3, :3].T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(view[:3, :3]), 1)
        projection = perspective(np.pi / 2, 2, 0.25, 20)
        for distance, expected_z in [(0.25, -1), (20, 1)]:
            clip = projection @ np.array([0, 0, -distance, 1])
            self.assertAlmostEqual(clip[2] / clip[3], expected_z)
        with self.assertRaises(ValueError):
            look_at([0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            perspective(np.pi / 2, 1, 2, 1)


if __name__ == "__main__":
    unittest.main()
