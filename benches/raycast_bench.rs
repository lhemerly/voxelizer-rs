use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use voxelizer_rs::MeshProcessor;

fn bench_raycast(c: &mut Criterion) {
    let mesh_processor = MeshProcessor::from_file("tests/data/cube.obj").unwrap();

    c.bench_function("voxelize_raycast", |b| {
        b.iter(|| mesh_processor.voxelize(black_box(0.1)));
    });
}

criterion_group!(benches, bench_raycast);
criterion_main!(benches);
