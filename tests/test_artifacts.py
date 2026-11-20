from pathlib import Path
import hashlib, json, csv, statistics
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_render_hashes_and_source_match(self):
        manifest = json.loads((ROOT / "results/image_sha256.json").read_text())
        self.assertEqual(len(manifest), 7)
        for name, digest in manifest.items():
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name
            )
        metadata = json.loads((ROOT / "results/metadata.json").read_text())
        self.assertEqual(
            hashlib.sha256((ROOT / "src/rasterlab/core.py").read_bytes()).hexdigest(),
            metadata["core_sha256"],
        )

    def test_diagnostics_and_timings_reconcile(self):
        checks = json.loads((ROOT / "results/diagnostics.json").read_text())
        self.assertEqual(checks["shared_edge_overlap_pixels"], 0)
        self.assertEqual(checks["shared_edge_covered_pixels"], 440 * 440)
        with (ROOT / "results/frame_timings.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(float(r["elapsed_ms"]) > 0 for r in rows))
        metadata = json.loads((ROOT / "results/metadata.json").read_text())
        self.assertAlmostEqual(
            statistics.median(float(r["elapsed_ms"]) for r in rows),
            metadata["median_frame_ms"],
        )

    def test_published_clipped_polygon_is_inside_frustum(self):
        data = json.loads((ROOT / "results/clipping.json").read_text())
        vertices = np.array(data["clip_output"])
        self.assertGreaterEqual(len(vertices), 3)
        self.assertTrue(np.all(np.abs(vertices[:, :3]) <= vertices[:, 3, None] + 1e-12))
        self.assertEqual(data["rasterized_fan_triangles"], len(vertices) - 2)


if __name__ == "__main__":
    unittest.main()
