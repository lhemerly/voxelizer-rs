## 2026-02-18 - Scanline Optimization Reversion in `parry3d`
**Learning:** Attempting to optimize `parry3d` raycasting by casting a single ray per line and counting intersection bounds to determine "inside" vs "outside" can fail when a mesh is un-closed, non-manifold, or has self-intersecting triangles (e.g. from generated STLs). The old logic raycasted exactly from the query point to determine `intersections % 2 != 0`. Caching intersections strictly across a scanline resulted in mismatched results compared to point-by-point raycasting. Additionally, reordering loops so X is inner and parallelizing over (Y, Z) is an effective performance boost on its own due to better spatial cache locality.
**Action:** Be extremely cautious optimizing geometric intersection algorithms on 3D meshes without verifying 100% equivalence, as mesh anomalies are frequent. When raycasting along an axis, parallelize over the orthogonal axes to maximize both Rayon granularity and CPU cache locality for the inner loop.

## 2024-04-19 - Pre-allocating vectors within tight inner loops

When Rayon `par_iter` spins up many small work units, frequent dynamic allocation of small `Vec`s (`Vec::new()`) in inner loops creates measurable overhead. By pre-allocating using `Vec::with_capacity(expected_len)`—especially when `expected_len` is statically known or easily bounded—we can reliably shave off ~5-6% of the execution time in performance-critical code paths without adding complexity.
## 2026-04-18 - Pre-allocated Vector Capacity in `load_obj`
**Learning:** Pre-calculating the total number of points and indices across all models in an OBJ file allows for using `Vec::with_capacity`. This optimization eliminates redundant reallocations and data copying as the vectors grow. For large meshes with numerous sub-models, this significantly reduces the "bookkeeping" overhead of dynamic arrays and minimizes memory fragmentation.
**Action:** Always check if the final size of a collection can be determined (even with a quick pre-pass) before populating it, especially in performance-critical data loading paths. Rationale for implementation without benchmarking: Vector reallocations are a well-known bottleneck in Rust/C++ when dealing with large datasets, and pre-allocation is a standard best practice with guaranteed (though sometimes small) performance gains.
## 2024-04-19 - Fast Scanline Raycasting
**Learning:** For solid voxelization, the naive approach casted a ray from EVERY voxel cell's center. By recognizing that we only need to query intersections along a single axis (e.g. +X), we can instead cast just ONE ray per (Y, Z) row starting from outside the mesh, record all intersection points, sort them, and then compute the parity (`intersections_to_right % 2 != 0`) for each cell along that scanline. This reduced the number of raycasts per row from `N` to `1`, resulting in an ~80% performance improvement for solid voxelization. While previous attempts at scanline caching had issues with un-closed meshes, sorting the intersections and computing exact intersections-to-the-right for each point perfectly matches the old point-by-point parity behavior.
**Action:** When performing grid-based intersection checks along a fixed axis, try to evaluate the entire line at once rather than independently per cell to drastically reduce BVH traversal overhead.
## 2024-05-19 - Inner Loop Optimizations

**Learning:** We attempted several optimizations in `MeshProcessor::voxelize`:
1. Pre-allocating `hit_xs` with `Vec::with_capacity(8)`.
2. Hoisting loop-invariant `base_x` calculation outside of the `ix` loops.
3. Hoisting loop-invariant `phase_sphere` parameters out of the innermost loops.
4. Replacing O(log N) `partition_point` with an O(1) amortized rolling index (`hit_idx`) for parity check.
5. Bypassing `self.mesh.distance_to_local_point()` BVH queries for exterior voxels when `narrow_band` is not used.

However, running `cargo bench` showed a performance regression of ~3-4% (from ~505µs to ~522µs) on a 10x10x10 cube test mesh. When benchmarking voxelization performance using small test meshes, asymptotic algorithmic improvements (such as O(log N) to O(1) lookups) may not show a measurable speedup. The overhead of setting up the rolling index and additional branching might have outweighed the benefits at this scale.

**Action:** Be aware that structural optimizations and algorithmic improvements might not show positive results on very small benchmark cases, and can even introduce slight regressions due to constant overhead. Evaluate structural optimizations using larger scales and higher voxel resolutions to capture their true performance impact.
