use assert_cmd::Command;

use std::fs;

#[test]
fn test_output_formats_contain_sdf() {
    let temp_dir = std::env::temp_dir();
    let csv_file = temp_dir.join("test_cube.csv");
    let ply_file = temp_dir.join("test_cube.ply");
    let vtk_file = temp_dir.join("test_cube.vtk");

    // Test CSV
    let mut cmd = Command::cargo_bin("voxelizer-rs").unwrap();
    cmd.args(&[
        "--input",
        "tests/data/cube.obj",
        "--output",
        csv_file.to_str().unwrap(),
        "--resolution",
        "0.5",
    ])
    .assert()
    .success();

    let csv_content = fs::read_to_string(&csv_file).unwrap();
    assert!(csv_content.contains("x,y,z,sdf,phase"));

    // Check for 5 columns in the first data row
    let first_data_row = csv_content.lines().nth(1).unwrap();
    assert_eq!(first_data_row.split(',').count(), 5);

    // Test PLY
    let mut cmd = Command::cargo_bin("voxelizer-rs").unwrap();
    cmd.args(&[
        "--input",
        "tests/data/cube.obj",
        "--output",
        ply_file.to_str().unwrap(),
        "--resolution",
        "0.5",
    ])
    .assert()
    .success();

    let ply_content = fs::read_to_string(&ply_file).unwrap();
    assert!(ply_content.contains("property float sdf"));

    // Check for 4 values in the first data row
    let header_lines = ply_content
        .lines()
        .filter(|l| !l.chars().next().unwrap().is_digit(10) && !l.starts_with('-'))
        .count();
    let first_data_row = ply_content.lines().nth(header_lines).unwrap();
    assert_eq!(first_data_row.split_whitespace().count(), 4);

    // Test VTK
    let mut cmd = Command::cargo_bin("voxelizer-rs").unwrap();
    cmd.args(&[
        "--input",
        "tests/data/cube.obj",
        "--output",
        vtk_file.to_str().unwrap(),
        "--resolution",
        "0.5",
    ])
    .assert()
    .success();

    let vtk_content = fs::read_to_string(&vtk_file).unwrap();
    assert!(vtk_content.contains("SCALARS sdf float 1"));

    // Clean up
    let _ = fs::remove_file(csv_file);
    let _ = fs::remove_file(ply_file);
    let _ = fs::remove_file(vtk_file);
}
