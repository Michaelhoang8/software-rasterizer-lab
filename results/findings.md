# Rasterization experiment results

The exact shared-edge square covers **193,600 pixels**, matching the analytic 440 × 440 area, with **0 double-covered pixels**. The pixel-center/top-left convention partitions the diagonal without gaps or overdraw.

The receding checker plane's affine and perspective-correct interpolation produce a mean UV-vector distance of 0.7446 texture-coordinate units over 137,406 mutually covered pixels. This describes the two methods' disagreement, not an image-quality score against a photographic reference. Analytic ray/plane tests establish which interpolation matches the projected surface.

The stored near-plane diagnostic clips one source triangle into a polygon before perspective division. Clip coordinates and the resulting polygon are committed in `clipping.json`; every vertex satisfies the six homogeneous half-space inequalities within floating-point tolerance.

The local CPU run measured a median **205.90 ms** for complete 480 × 320 hero-frame generation across 12 samples, after one warmup. The timed path includes mesh/texture generation, transformations, clipping, rasterization and shading but excludes PNG encoding. This is not a GPU or production-engine benchmark; timing varies with hardware and background load.

Recorded 2026-11-20T22:46:28.574016+00:00; Python 3.13.17, NumPy 2.3.5, Windows AMD64. Raw samples and source hash are committed.
