# Voxelizer-rs

A high-performance, parallelized CLI tool written in Rust that converts 3D meshes (STL/OBJ) into particle clouds. 

## Features

### Fast: 
Uses rayon for multi-threaded ray-casting and grid search.

### Robust: Handles both .stl (binary/ascii) and .obj formats.

### Multi-format Export: Outputs to PLY, CSV, VTK, OBJ, or a compact binary (.bin) format depending on the file extension.

### Surface and Solid Modes: Generate either a solid voxelized volume, or just a thin surface shell.

## Installation

```
cargo install voxelizer-rs
```

## Usage

```
voxelizer-rs --input heart_model.stl --output heart.bin --resolution 0.5
```

Arguments:

``` -i, --input ```: Path to source mesh. 

``` -o, --output ```: Path to destination file. The format is inferred from the extension (`.csv`, `.ply`, `.vtk`, `.obj`, or `.bin`).

``` -r, --resolution ```: Voxel size in mesh units (default: 0.5).

``` --surface-only ```: Optional flag. If provided, the output will only contain surface voxels instead of solid filling.

``` --narrow-band ```: Optional flag. Evaluate and store only the voxels where the absolute distance is within the desired threshold.

``` --scale ```: Scale multiplier for the mesh dimensions (default: 1.0).

``` --center ```: Optional flag. If provided, the mesh will be centered at the origin (0, 0, 0) before processing.

``` --translate ```: Translates the mesh by 'x,y,z' offsets (e.g., `--translate 10,-5,0`).

``` --rotate ```: Rotates the mesh by 'x,y,z' degrees (e.g., `--rotate 90,0,0`).

``` --crop ```: Crops the mesh to the given bounding box 'min_x,min_y,min_z,max_x,max_y,max_z' (e.g., `--crop -10,-10,-10,10,10,10`).

``` --vertex-noise ```: Adds random displacement noise with the specified amplitude.

``` --phase-sphere ```: Specify a sphere for assigning a phase 'x,y,z,radius' (e.g., `--phase-sphere 0,0,0,5`).

``` --threads ```: Number of parallel threads to use. Defaults to the system's logical core count.

## Binary Format Specification

The output file uses a compact custom binary format serialized via ```bincode```.

### 1. Header (Little Endian)

| Offset | Type | Description |
|---|---|---|
| 0x00 | u32 | Version (Current: 2) |
| 0x04 | u64 | Particle Count (N) |
| 0x0C | f64 | Resolution (Grid spacing) |

### 2. Payload (Little Endian)

The header is immediately followed by a 64-bit length indicator (part of the bincode vector serialization), and then N particle structs.

Layout per Particle (32 bytes):

```
struct ParticleData {
    x: f32,
    y: f32,
    z: f32,
    sdf: f32,
    phase: u32, // 0 = Fluid, 1 = Wall/Solid
    label_id: u32,
    fiber_x: f32,
    fiber_y: f32,
}
```