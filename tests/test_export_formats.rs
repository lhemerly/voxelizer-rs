use std::fs;
use std::process::Command;
use parry3d::shape::Ball;

#[test]
fn test_xyz_export() {
    let radius = 2.0;
    let ball = Ball::new(radius);
    let (vertices, indices) = ball.to_trimesh(10, 10);

    let faces: Vec<[f32; 3]> = vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let mut stl_faces = Vec::new();
    for idx in &indices {
        let v0 = faces[idx[0] as usize];
        let v1 = faces[idx[1] as usize];
        let v2 = faces[idx[2] as usize];
        stl_faces.push(vec![v0, v1, v2]);
    }

    let temp_dir = std::env::temp_dir();
    let stl_path = temp_dir.join(format!(
        "test_sphere_export_{}.stl",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));

    let mut f = std::fs::File::create(&stl_path).unwrap();
    use std::io::Write;
    f.write_all(&[0; 80]).unwrap();
    f.write_all(&(stl_faces.len() as u32).to_le_bytes()).unwrap();
    for v in &stl_faces {
        f.write_all(&[0; 12]).unwrap();
        for pt in v {
            for c in pt {
                f.write_all(&c.to_le_bytes()).unwrap();
            }
        }
        f.write_all(&[0; 2]).unwrap();
    }

    let xyz_path = temp_dir.join(format!(
        "test_output_{}.xyz",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));

    let cargo_bin = std::env::var("CARGO_BIN_EXE_voxelizer-rs")
        .unwrap_or_else(|_| "target/debug/voxelizer-rs".to_string());

    let output = Command::new(&cargo_bin)
        .arg("--input")
        .arg(stl_path.to_str().unwrap())
        .arg("--output")
        .arg(xyz_path.to_str().unwrap())
        .arg("--resolution")
        .arg("1.0")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "CLI command failed: {:?}", output);

    let contents = fs::read_to_string(&xyz_path).expect("Failed to read XYZ file");
    assert!(!contents.is_empty(), "XYZ file is empty");

    // An XYZ file should have one point per line, space separated coordinates
    let lines: Vec<&str> = contents.trim().split('\n').collect();
    assert!(!lines.is_empty());

    for line in lines {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        assert_eq!(parts.len(), 3, "Expected 3 coordinates per line, got {:?}", parts);
        let x: f32 = parts[0].parse().unwrap();
        let y: f32 = parts[1].parse().unwrap();
        let z: f32 = parts[2].parse().unwrap();
        // Since it's a sphere centered at origin, bounding box coords should be roughly within [-2, 2]
        assert!(x >= -3.0 && x <= 3.0);
        assert!(y >= -3.0 && y <= 3.0);
        assert!(z >= -3.0 && z <= 3.0);
    }

    let _ = fs::remove_file(stl_path);
    let _ = fs::remove_file(xyz_path);
}
