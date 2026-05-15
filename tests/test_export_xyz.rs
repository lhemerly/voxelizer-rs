use std::process::Command;

use std::fs;

#[test]
fn test_export_xyz() {
    let mut bin_path = std::env::current_exe().unwrap();
    bin_path.pop(); // deps
    bin_path.pop(); // debug
    bin_path.push("voxelizer-rs");

    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join(format!(
        "test_export_{}.xyz",
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ));

    // Run the voxelizer tool on the test cube
    let output = Command::new(bin_path)
        .arg("-i")
        .arg("tests/data/cube.obj")
        .arg("-o")
        .arg(&output_path)
        .arg("-r")
        .arg("1.0")
        .output()
        .expect("Failed to execute voxelizer");

    assert!(output.status.success(), "Command failed: {:?}", output);

    // Read the file and assert its content
    let content = fs::read_to_string(&output_path).unwrap();
    assert_eq!(content.trim(), "0.5 0.5 0.5");

    // Clean up
    fs::remove_file(output_path).unwrap();
}
