# Rasterization pipeline

## Coordinate conventions

Positions use column-vector mathematics with `clip = projection × view × model × position`. NumPy batches are stored as rows and multiplied by the transposed matrix. The view is right-handed and looks along −Z. Projection maps the visible frustum to `−w ≤ x, y, z ≤ w`, with positive w for ordinary visible points.

NDC follows perspective division. Screen coordinates map x from [−1,1] to [0,width], and invert y into a top-left-origin image. Pixel centers are `(x+0.5,y+0.5)`. NDC z in [−1,1] maps linearly to window depth [0,1], where smaller is nearer.

## Homogeneous clipping

Six plane equations represent the frustum: `w+x`, `w−x`, `w+y`, `w−y`, `w+z`, `w−z`. A point is inside a plane when its signed plane value is nonnegative.

For an edge with signed distances d₀ and d₁, a crossing uses `t = d₀ / (d₀−d₁)`. Clip positions, UVs and world normals are interpolated with that same t **before** perspective division. Sutherland–Hodgman visits each plane in order. The surviving convex polygon is triangulated as a fan.

Tests cross every plane independently, check generated attributes and use unequal homogeneous w values to distinguish correct clipping from incorrect post-divide interpolation. Degenerate zero-area triangles contribute no coverage. Invalid nonfinite clip vertices are rejected.

## Edge functions and top-left ownership

The edge function is `E(a,b,p) = (b.x−a.x)(p.y−a.y) − (b.y−a.y)(p.x−a.x)`. Triangle winding is normalized to positive screen-space area. Opposite-edge values divided by total area provide barycentric weights.

For the y-down convention, an edge is top-left when `dy < 0`, or when `dy == 0` and `dx > 0`. A sample is inside if every edge is positive, or exactly zero on an inclusive top-left edge. The pixel bounding box is clamped before allocation/indexing, preventing offscreen wrapping.

This implementation uses float64 and exact zero comparisons. Exact shared vertices in the diagnostic partition cleanly under either diagonal and reversed winding. Arbitrary transformed/clipped adjacent polygons may acquire slightly different floating-point endpoints. The implementation does not claim fixed-point GPU-style watertightness for all geometry.

## Attributes versus depth

For screen-space barycentrics λ, perspective-correct attribute interpolation is:

```text
a = (Σ λᵢ aᵢ / wᵢ) / (Σ λᵢ / wᵢ)
```

UVs and world normals use this relation. Normals transform using the inverse transpose of the model's linear transform, then are normalized after interpolation. Nearest texture sampling repeats coordinates modulo one.

Depth follows a different rule: interpolate each vertex's **z/w affinely** using screen barycentrics, then map to [0,1]. Applying reciprocal-w interpolation again to NDC depth would be incorrect. The tests derive expected UVs and depth independently from a ray intersecting the original 3D triangle plane.

The depth test is strict `<`; equal-depth samples keep the first submission. Draw-order invariance tests concern distinct depths, not coplanar ties.

## Shading and scene generation

The torus is generated parametrically with separate major/tube segments, texture coordinates and vertex normals. The hero uses 64 × 32 segments (4,096 torus triangles) and a two-triangle ground plane. The movie uses 40 × 20 segments to reduce CPU cost. The timing scene uses 32 × 16 segments.

Directional Lambert-like shading, a small stylized rim term and a background gradient make the surface orientation readable. They are not a physically based rendering model. No ray tracer or 3D plotting library produces the images. Pillow only encodes the completed RGB framebuffer.
