# Reproduction and measurement

The render script produces a procedural torus scene; affine/corrected UV plane images; a near-plane crossing; exact shared-edge ownership; overlapping triangles; and a false-color depth buffer. `results/clipping.json` records input and surviving homogeneous vertices. `results/diagnostics.json` records coverage counts and UV-method disagreement.

UV disagreement is the Euclidean distance between interpolated coordinates, averaged over pixels covered in both images. It is expressed in texture-coordinate units before wrapping. It is not PSNR, a photographic quality metric or proof of visual correctness by itself. Independent analytic tests establish the correct projective mapping.

## Timing

One untimed warmup precedes 12 complete frames at 480 × 320. The same camera, angle, textures and geometry are used in every sample. Timing uses `perf_counter_ns`. The path includes procedural mesh/texture construction, transforms, clipping, bounding boxes, coverage, z testing and shading. It excludes PNG encoding and FFmpeg. The quick CI run uses three samples.

`frame_timings.csv` retains sample durations and triangle counts; `metadata.json` records the median, Python/NumPy versions, OS/architecture and source hash. No CPU affinity, cache-flush policy or system isolation is used. Other workloads and machines will differ. The animation is encoded at 12 fps regardless of how slowly the CPU produced its frames; its playback rate is not real-time renderer performance.

## Determinism

Scene generation is deterministic; no random source assets are used. Checked-in PNG hashes verify the published image bytes. Small cross-platform floating-point differences or different PNG encoders can change regenerated hashes, so CI tests analytic semantics and successful reproduction instead of requiring every new PNG to match a single platform's bytes.

The original hash checks still protect the checked-in images from accidental alteration. Git attributes preserve binary image/video files and LF source files. Timing and run timestamps naturally change on every experiment.
