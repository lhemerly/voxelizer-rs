use bincode;
use voxelizer_rs::ParticleHeader;

#[test]
fn test_layout() {
    let header = ParticleHeader {
        version: 2,
        particle_count: 100,
        resolution: 0.5,
    };
    let bytes = bincode::serialize(&header).unwrap();
    println!("Bytes: {:?}", bytes);
    assert_eq!(bytes.len(), 20); // 4 + 8 + 8 = 20
}
