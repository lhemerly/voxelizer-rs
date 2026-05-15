## 2024-05-18 - Optimized mesh transformation pass
**Learning:** Consolidating multiple memory passes (rotation, centering, scaling, translation, noise) into two passes and manually tracking bounds during the final pass provides significant performance gains for large point clouds by reducing memory traffic and avoiding the O(N) `TriMesh::local_aabb()` calculation.
**Action:** Refactored `MeshProcessor::from_file` in `src/lib.rs` to consolidate transformations and bounds calculation.
