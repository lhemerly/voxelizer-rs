#!/bin/bash
sed -i 's/\.voxelize(res, false, None, None)/.voxelize(res, false, None, None, None)/g' src/lib.rs
sed -i 's/\.voxelize(0.5, false, None, None)/.voxelize(0.5, false, None, None, None)/g' src/lib.rs
sed -i 's/\.voxelize(0.5, false, Some(band), None)/.voxelize(0.5, false, Some(band), None, None)/g' src/lib.rs
sed -i 's/\.voxelize(0.5, false, Some(0.0), None)/.voxelize(0.5, false, Some(0.0), None, None)/g' src/lib.rs
sed -i 's/\.voxelize(0.5, false, Some(2.0), None)/.voxelize(0.5, false, Some(2.0), None, None)/g' src/lib.rs
sed -i 's/\.voxelize(resolution, false, narrow_band, None)/.voxelize(resolution, false, narrow_band, None, None)/g' tests/test_sdf_invariants.rs
sed -i 's/\.voxelize(black_box(0.1), false, None, None)/.voxelize(black_box(0.1), false, None, None, None)/g' benches/voxelize_bench.rs
