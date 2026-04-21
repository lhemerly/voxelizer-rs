use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use voxelizer_rs::MeshProcessor;

fn bench_load_obj(c: &mut Criterion) {
    c.bench_function("load_obj", |b| {
        b.iter(|| MeshProcessor::from_file(black_box("tests/data/cube.obj")));
    });
}

criterion_group!(benches, bench_load_obj);
criterion_main!(benches);
