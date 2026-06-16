use criterion::{Criterion, black_box, criterion_group, criterion_main};
use voxelizer_rs::{MeshProcessor, TransformConfig};

fn bench_voxelize(c: &mut Criterion) {
    let transform = TransformConfig::default();
    let mesh_processor = MeshProcessor::from_file("tests/data/cube.obj", &transform).unwrap();

    c.bench_function("voxelize", |b| {
        b.iter(|| mesh_processor.voxelize(black_box(0.1), false, None, None, None));
    });
}

criterion_group!(benches, bench_voxelize);
criterion_main!(benches);
