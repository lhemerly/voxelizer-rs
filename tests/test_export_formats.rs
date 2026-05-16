use assert_cmd::Command;
use std::fs;
use tempfile::NamedTempFile;

#[test]
fn test_export_xyz() {
    let output_file = NamedTempFile::new().unwrap();
    let output_path = output_file.path().with_extension("xyz");

    let mut cmd = Command::cargo_bin("voxelizer-rs").unwrap();
    cmd.arg("--input")
        .arg("tests/data/cube.obj")
        .arg("--output")
        .arg(&output_path)
        .arg("--resolution")
        .arg("0.5")
        .assert()
        .success();

    let content = fs::read_to_string(&output_path).unwrap();
    assert!(!content.is_empty());

    // XYZ should just have x y z separated by spaces
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() > 0);
    let parts: Vec<&str> = lines[0].split_whitespace().collect();
    assert_eq!(parts.len(), 3);

    // Clean up
    let _ = fs::remove_file(output_path);
}
