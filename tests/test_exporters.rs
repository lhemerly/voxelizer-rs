use assert_cmd::Command;
use std::fs;

#[test]
fn test_csv_exporter_includes_sdf() {
    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join(format!(
        "test_cube_{}.obj",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));
    let output_path = temp_dir.join(format!(
        "test_output_{}.csv",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));

    fs::write(
        &input_path,
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3",
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("voxelizer-rs").unwrap();
    cmd.args(&[
        "--input",
        input_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--resolution",
        "0.5",
    ]);
    cmd.assert().success();

    let contents = fs::read_to_string(&output_path).unwrap();
    assert!(contents.starts_with("x,y,z,sdf,phase\n"));

    fs::remove_file(input_path).unwrap();
    fs::remove_file(output_path).unwrap();
}

#[test]
fn test_ply_exporter_includes_sdf() {
    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join(format!(
        "test_cube_{}.obj",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));
    let output_path = temp_dir.join(format!(
        "test_output_{}.ply",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));

    fs::write(
        &input_path,
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3",
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("voxelizer-rs").unwrap();
    cmd.args(&[
        "--input",
        input_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--resolution",
        "0.5",
    ]);
    cmd.assert().success();

    let contents = fs::read_to_string(&output_path).unwrap();
    assert!(contents.contains("property float sdf"));

    fs::remove_file(input_path).unwrap();
    fs::remove_file(output_path).unwrap();
}

#[test]
fn test_vtk_exporter_includes_sdf() {
    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join(format!(
        "test_cube_{}.obj",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));
    let output_path = temp_dir.join(format!(
        "test_output_{}.vtk",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));

    fs::write(
        &input_path,
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3",
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("voxelizer-rs").unwrap();
    cmd.args(&[
        "--input",
        input_path.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
        "--resolution",
        "0.5",
    ]);
    cmd.assert().success();

    let contents = fs::read_to_string(&output_path).unwrap();
    assert!(contents.contains("SCALARS sdf float 1"));

    fs::remove_file(input_path).unwrap();
    fs::remove_file(output_path).unwrap();
}
