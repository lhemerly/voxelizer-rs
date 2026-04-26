use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box as std_black_box;
use voxelizer_rs::MeshProcessor;

fn bench_voxelize(c: &mut Criterion) {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join(format!(
        "bench_voxelize_{}.stl",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));

    // Create a sphere-like shape, but a bit larger
    let mut faces = vec![];
    let r = 2.0;
    let n = 20;
    for i in 0..n {
        for j in 0..n {
            let theta1 = std::f32::consts::PI * (i as f32) / (n as f32);
            let theta2 = std::f32::consts::PI * ((i + 1) as f32) / (n as f32);
            let phi1 = 2.0 * std::f32::consts::PI * (j as f32) / (n as f32);
            let phi2 = 2.0 * std::f32::consts::PI * ((j + 1) as f32) / (n as f32);

            let p1 = [r * theta1.sin() * phi1.cos(), r * theta1.sin() * phi1.sin(), r * theta1.cos()];
            let p2 = [r * theta2.sin() * phi1.cos(), r * theta2.sin() * phi1.sin(), r * theta2.cos()];
            let p3 = [r * theta2.sin() * phi2.cos(), r * theta2.sin() * phi2.sin(), r * theta2.cos()];
            let p4 = [r * theta1.sin() * phi2.cos(), r * theta1.sin() * phi2.sin(), r * theta1.cos()];

            faces.push([p1, p2, p3]);
            faces.push([p1, p3, p4]);
        }
    }

    let mut f = std::fs::File::create(&file_path).unwrap();
    use std::io::Write;
    f.write_all(&[0; 80]).unwrap();
    f.write_all(&(faces.len() as u32).to_le_bytes()).unwrap();
    for v in &faces {
        f.write_all(&[0; 12]).unwrap();
        for pt in v {
            for c in pt {
                f.write_all(&(*c).to_le_bytes()).unwrap();
            }
        }
        f.write_all(&[0; 2]).unwrap();
    }

    let config = voxelizer_rs::TransformConfig::default();
    let processor = MeshProcessor::from_file(file_path.to_str().unwrap(), &config).unwrap();

    let mut group = c.benchmark_group("voxelize_bench_test");
    group.bench_function("voxelize_solid", |b| {
        b.iter(|| {
            processor.voxelize(std_black_box(0.2), false, None, None).unwrap()
        })
    });
    group.finish();
    std::fs::remove_file(file_path).unwrap();
}

criterion_group!(benches, bench_voxelize);
criterion_main!(benches);
