## 2024-05-23 - Optimize voxelization loop using hoisted calculations and rolling scanline intersections

**Learning:** Voxelization benchmark results using small 10x10x10 cubes mask the impact of asymptotic algorithmic improvements, like avoiding `hit_xs.partition_point` O(log N) lookups in favor of an O(1) rolling index inside the innermost loop. While `cargo bench` reports changes within noise threshold (+1.21%) on these small meshes, pre-calculating values (such as `base_x` and invariant `phase_sphere` factors) avoids thousands of redundant loop-invariant operations and will scale much better for real-world large models where mesh complexities make operations highly expensive. Furthermore, exterior voxels can quickly be discarded without performing expensive `distance_to_local_point()` queries on the BVH tree.

**Action:** Refactored `MeshProcessor::voxelize` inside `src/lib.rs` by:
1. Hoisting arithmetic operations like base coordinates and sphere invariant calculations outside the parallel raycasting loops.
2. Initializing vectors (`hit_xs`) with an adequate initial capacity `Vec::with_capacity(8)`.
3. Adopting a rolling index `hit_idx` to iterate over sorted scanline hit points (exploiting the strictly ascending X-coordinates in the innermost loop), achieving an amortized O(1) inside/outside test instead of repeated O(log N) bounds lookups.
4. Short-circuiting evaluation for known exterior voxels (when `narrow_band` is not used) without running a full `parry3d` distance query.