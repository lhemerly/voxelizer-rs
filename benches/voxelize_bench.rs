use criterion::{Criterion, criterion_group, criterion_main, black_box};
use voxelizer_rs::MeshProcessor;

fn bench_voxelize(c: &mut Criterion) {
    let mesh_processor = MeshProcessor::from_file("tests/data/cube.obj").unwrap();
    c.bench_function("voxelize", |b| {
        b.iter(|| mesh_processor.voxelize(black_box(0.1)));
    });
}

criterion_group!(benches, bench_voxelize);
criterion_main!(benches);
