use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use voxelizer_rs::{MeshProcessor, ParticleHeader, TransformConfig};

#[derive(Parser, Debug)]
#[command(author, version, about = "Voxelizer")]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,

    #[arg(short, long, default_value_t = 0.5, value_parser = validate_resolution)]
    resolution: f64,

    #[arg(long)]
    surface_only: bool,

    #[arg(long, value_parser = validate_narrow_band)]
    narrow_band: Option<f64>,

    #[arg(long, default_value_t = 1.0)]
    scale: f64,

    #[arg(long)]
    center: bool,

    #[arg(long, value_parser = parse_vec3)]
    translate: Option<[f64; 3]>,

    #[arg(long, value_parser = parse_vec3)]
    rotate: Option<[f64; 3]>,

    #[arg(long, value_parser = parse_vec6)]
    crop: Option<[f64; 6]>,

    #[arg(long)]
    vertex_noise: Option<f64>,

    #[arg(long, value_parser = parse_vec4)]
    phase_sphere: Option<[f64; 4]>,

    #[arg(long)]
    sparsify: Option<f64>,

    #[arg(long)]
    hollow: Option<f64>,

    #[arg(long)]
    lattice_spacing: Option<u64>,

    #[arg(long, default_value_t = 1)]
    lattice_thickness: u64,

    #[arg(long)]
    export_slices: Option<String>,

    #[arg(long)]
    threads: Option<usize>,
}

fn parse_vec4(s: &str) -> Result<[f64; 4], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err(format!("Expected 'x,y,z,radius', got '{}'", s));
    }
    let x = parts[0]
        .parse()
        .map_err(|_| format!("Invalid x: {}", parts[0]))?;
    let y = parts[1]
        .parse()
        .map_err(|_| format!("Invalid y: {}", parts[1]))?;
    let z = parts[2]
        .parse()
        .map_err(|_| format!("Invalid z: {}", parts[2]))?;
    let w = parts[3]
        .parse()
        .map_err(|_| format!("Invalid radius: {}", parts[3]))?;
    Ok([x, y, z, w])
}

fn parse_vec6(s: &str) -> Result<[f64; 6], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 6 {
        return Err(format!(
            "Expected 'min_x,min_y,min_z,max_x,max_y,max_z', got '{}'",
            s
        ));
    }
    let v0 = parts[0]
        .parse()
        .map_err(|_| format!("Invalid value: {}", parts[0]))?;
    let v1 = parts[1]
        .parse()
        .map_err(|_| format!("Invalid value: {}", parts[1]))?;
    let v2 = parts[2]
        .parse()
        .map_err(|_| format!("Invalid value: {}", parts[2]))?;
    let v3 = parts[3]
        .parse()
        .map_err(|_| format!("Invalid value: {}", parts[3]))?;
    let v4 = parts[4]
        .parse()
        .map_err(|_| format!("Invalid value: {}", parts[4]))?;
    let v5 = parts[5]
        .parse()
        .map_err(|_| format!("Invalid value: {}", parts[5]))?;
    Ok([v0, v1, v2, v3, v4, v5])
}

fn parse_vec3(s: &str) -> Result<[f64; 3], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("Expected 'x,y,z', got '{}'", s));
    }
    let x = parts[0]
        .parse()
        .map_err(|_| format!("Invalid x: {}", parts[0]))?;
    let y = parts[1]
        .parse()
        .map_err(|_| format!("Invalid y: {}", parts[1]))?;
    let z = parts[2]
        .parse()
        .map_err(|_| format!("Invalid z: {}", parts[2]))?;
    Ok([x, y, z])
}

fn validate_resolution(s: &str) -> Result<f64, String> {
    let val: f64 = s.parse().map_err(|_| format!("`{s}` isn't a number"))?;
    if val.is_finite() && val > 1e-6 {
        Ok(val)
    } else {
        Err(format!(
            "Resolution must be a finite number greater than 1e-6. Provided: {s}"
        ))
    }
}

fn validate_narrow_band(s: &str) -> Result<f64, String> {
    let val: f64 = s.parse().map_err(|_| format!("`{s}` isn't a number"))?;
    if val.is_finite() && val >= 0.0 {
        Ok(val)
    } else {
        Err(format!(
            "Narrow band must be a finite non-negative number. Provided: {s}"
        ))
    }
}

fn export_pbm_slices(
    particles: &[voxelizer_rs::ParticleData],
    output_dir: &str,
    res: f64,
) -> anyhow::Result<()> {
    if particles.is_empty() {
        return Ok(());
    }

    std::fs::create_dir_all(output_dir)?;

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for p in particles {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        min_z = min_z.min(p.z);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }

    let res_f32 = res as f32;
    let size_x = ((max_x - min_x) / res_f32).round() as usize + 1;
    let size_y = ((max_y - min_y) / res_f32).round() as usize + 1;

    let mut layers: std::collections::HashMap<i32, std::collections::HashSet<(usize, usize)>> =
        std::collections::HashMap::new();

    for p in particles {
        let vx = ((p.x - min_x) / res_f32).round() as usize;
        let vy = ((p.y - min_y) / res_f32).round() as usize;
        let vz = ((p.z - min_z) / res_f32).round() as i32;

        layers.entry(vz).or_default().insert((vx, vy));
    }

    let mut z_indices: Vec<i32> = layers.keys().copied().collect();
    z_indices.sort_unstable();

    for (layer_idx, z) in z_indices.into_iter().enumerate() {
        let points = &layers[&z];
        let file_path = Path::new(output_dir).join(format!("slice_{:04}.pbm", layer_idx));
        let mut f = BufWriter::new(File::create(&file_path)?);

        writeln!(f, "P1")?;
        writeln!(f, "{} {}", size_x, size_y)?;

        // PBM coordinates: y is usually from top to bottom.
        // We'll iterate y from 0 to size_y - 1 (which corresponds to min_y to max_y)
        for y in 0..size_y {
            for x in 0..size_x {
                if points.contains(&(x, y)) {
                    write!(f, "1 ")?;
                } else {
                    write!(f, "0 ")?;
                }
            }
            writeln!(f)?;
        }
    }

    println!("Exported {} slices to '{}'", layers.len(), output_dir);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| anyhow::anyhow!("Failed to configure thread pool: {}", e))?;
    }

    println!("Input: {}", args.input);
    println!("Resolution: {} mm", args.resolution);

    let transform = TransformConfig {
        scale: args.scale,
        center: args.center,
        translate: args.translate,
        rotate: args.rotate,
        crop: args.crop,
        vertex_noise: args.vertex_noise,
    };

    let opts = voxelizer_rs::VoxelizeOptions {
        resolution: args.resolution,
        surface_only: args.surface_only,
        narrow_band: args.narrow_band,
        phase_sphere: args.phase_sphere,
        sparsify: args.sparsify,
        hollow: args.hollow,
        lattice_spacing: args.lattice_spacing,
        lattice_thickness: args.lattice_thickness,
    };

    let processor = MeshProcessor::from_file(&args.input, &transform)?;
    let particles = processor.voxelize(&opts)?;

    println!("Generated {} particles.", particles.len());

    if let Some(slice_dir) = args.export_slices {
        export_pbm_slices(&particles, &slice_dir, args.resolution)?;
    }

    let path_out = Path::new(&args.output);
    let extension = path_out
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.to_lowercase());

    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);

    match extension.as_deref() {
        Some("csv") => {
            writeln!(writer, "x,y,z,phase")?;
            for p in &particles {
                writeln!(writer, "{},{},{},{}", p.x, p.y, p.z, p.phase)?;
            }
        }
        Some("ply") => {
            writeln!(writer, "ply")?;
            writeln!(writer, "format ascii 1.0")?;
            writeln!(writer, "element vertex {}", particles.len())?;
            writeln!(writer, "property float x")?;
            writeln!(writer, "property float y")?;
            writeln!(writer, "property float z")?;
            writeln!(writer, "end_header")?;
            for p in &particles {
                writeln!(writer, "{} {} {}", p.x, p.y, p.z)?;
            }
        }
        Some("vtk") => {
            writeln!(writer, "# vtk DataFile Version 3.0")?;
            writeln!(writer, "Voxelizer Output")?;
            writeln!(writer, "ASCII")?;
            writeln!(writer, "DATASET POLYDATA")?;
            writeln!(writer, "POINTS {} float", particles.len())?;
            for p in &particles {
                writeln!(writer, "{} {} {}", p.x, p.y, p.z)?;
            }
            writeln!(writer, "POINT_DATA {}", particles.len())?;
            writeln!(writer, "SCALARS phase int 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for p in &particles {
                writeln!(writer, "{}", p.phase)?;
            }
        }
        Some("vox") => {
            if particles.is_empty() {
                println!("No particles to export. Cannot generate empty VOX file.");
                anyhow::bail!("Cannot generate empty VOX file.");
            }

            let mut min_x = f32::MAX;
            let mut min_y = f32::MAX;
            let mut min_z = f32::MAX;
            let mut max_x = f32::MIN;
            let mut max_y = f32::MIN;
            let mut max_z = f32::MIN;

            for p in &particles {
                if p.x < min_x {
                    min_x = p.x;
                }
                if p.y < min_y {
                    min_y = p.y;
                }
                if p.z < min_z {
                    min_z = p.z;
                }
                if p.x > max_x {
                    max_x = p.x;
                }
                if p.y > max_y {
                    max_y = p.y;
                }
                if p.z > max_z {
                    max_z = p.z;
                }
            }

            let res = args.resolution as f32;
            let mut size_x = ((max_x - min_x) / res).round() as u32 + 1;
            let mut size_y = ((max_y - min_y) / res).round() as u32 + 1;
            let mut size_z = ((max_z - min_z) / res).round() as u32 + 1;

            if size_x > 256 || size_y > 256 || size_z > 256 {
                println!(
                    "Warning: The voxel grid ({size_x}x{size_y}x{size_z}) exceeds MagicaVoxel's 256x256x256 per-model limit. Voxels outside this range will be clamped or wrapped."
                );
                size_x = size_x.min(256);
                size_y = size_y.min(256);
                size_z = size_z.min(256);
            }

            // MagicaVoxel format requires a 256-color palette. We'll just use color index 1 for solid and 2 for surface.
            // Since phase isn't always reliable for surface/solid coloring, we can color by phase or just use 1.
            let num_voxels = particles.len() as u32;

            // Header
            writer.write_all(b"VOX ")?;
            writer.write_all(&150u32.to_le_bytes())?;

            // MAIN chunk
            writer.write_all(b"MAIN")?;
            writer.write_all(&0u32.to_le_bytes())?;
            // Size of children: SIZE chunk (24) + XYZI chunk (16 + 4 * num_voxels)
            let children_size = 24 + 16 + num_voxels * 4;
            writer.write_all(&children_size.to_le_bytes())?;

            // SIZE chunk
            writer.write_all(b"SIZE")?;
            writer.write_all(&12u32.to_le_bytes())?;
            writer.write_all(&0u32.to_le_bytes())?;
            writer.write_all(&size_x.to_le_bytes())?;
            writer.write_all(&size_y.to_le_bytes())?;
            writer.write_all(&size_z.to_le_bytes())?;

            // XYZI chunk
            writer.write_all(b"XYZI")?;
            let xyzi_content_size = 4 + num_voxels * 4;
            writer.write_all(&xyzi_content_size.to_le_bytes())?;
            writer.write_all(&0u32.to_le_bytes())?;
            writer.write_all(&num_voxels.to_le_bytes())?;

            for p in &particles {
                let mut vx = ((p.x - min_x) / res).round() as i32;
                let mut vy = ((p.y - min_y) / res).round() as i32;
                let mut vz = ((p.z - min_z) / res).round() as i32;

                // Clamp to [0, 255]
                vx = vx.clamp(0, 255);
                vy = vy.clamp(0, 255);
                vz = vz.clamp(0, 255);

                let color_idx = if p.phase > 0 { 2u8 } else { 1u8 };
                writer.write_all(&[vx as u8, vy as u8, vz as u8, color_idx])?;
            }
        }
        Some("obj") => {
            for p in &particles {
                writeln!(writer, "v {} {} {}", p.x, p.y, p.z)?;
            }
        }
        _ => {
            // Default to BIN
            let header = ParticleHeader {
                version: 2,
                particle_count: particles.len() as u64,
                resolution: args.resolution,
            };

            bincode::serialize_into(&mut writer, &header)?;
            bincode::serialize_into(&mut writer, &particles)?;
        }
    }

    println!("Saved to {}", args.output);
    Ok(())
}
