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
    threads: Option<usize>,

    #[arg(long)]
    preview: bool,

    #[arg(long)]
    disintegrate: Option<f64>,

    #[arg(long, value_parser = parse_vec2)]
    wave: Option<[f64; 2]>,
}

fn parse_vec2(s: &str) -> Result<[f64; 2], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("Expected 'x,y', got '{}'", s));
    }
    let x = parts[0]
        .parse()
        .map_err(|_| format!("Invalid x: {}", parts[0]))?;
    let y = parts[1]
        .parse()
        .map_err(|_| format!("Invalid y: {}", parts[1]))?;
    Ok([x, y])
}

fn print_preview(particles: &[voxelizer_rs::ParticleData], _resolution: f64) {
    if particles.is_empty() {
        return;
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut max_z = f32::MIN;

    for p in particles {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        min_z = min_z.min(p.z);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
        max_z = max_z.max(p.z);
    }

    let width = 60;
    let height = 30;

    // We'll generate two views: Top (X-Z) and Front (X-Y)

    let draw_view = |get_x: &dyn Fn(&voxelizer_rs::ParticleData) -> f32,
                     get_y: &dyn Fn(&voxelizer_rs::ParticleData) -> f32,
                     view_min_x: f32,
                     view_max_x: f32,
                     view_min_y: f32,
                     view_max_y: f32|
     -> Vec<String> {
        let span_x = (view_max_x - view_min_x).max(1e-5);
        let span_y = (view_max_y - view_min_y).max(1e-5);

        // Braille characters represent a 2x4 pixel grid.
        let pixel_width = width * 2;
        let pixel_height = height * 4;

        let mut grid = vec![false; pixel_width * pixel_height];

        for p in particles {
            let px = get_x(p);
            let py = get_y(p);

            let x = (((px - view_min_x) / span_x) * (pixel_width as f32 - 1.0)).round() as usize;
            // Invert Y so up is up in the terminal
            let y = (((view_max_y - py) / span_y) * (pixel_height as f32 - 1.0)).round() as usize;

            if x < pixel_width && y < pixel_height {
                grid[y * pixel_width + x] = true;
            }
        }

        let mut lines = Vec::new();
        for by in 0..height {
            let mut line = String::new();
            for bx in 0..width {
                let mut v = 0;
                for dy in 0..4 {
                    for dx in 0..2 {
                        let px = bx * 2 + dx;
                        let py = by * 4 + dy;
                        if grid[py * pixel_width + px] {
                            // Braille dot mapping
                            let dot_idx = match (dx, dy) {
                                (0, 0) => 0,
                                (0, 1) => 1,
                                (0, 2) => 2,
                                (1, 0) => 3,
                                (1, 1) => 4,
                                (1, 2) => 5,
                                (0, 3) => 6,
                                (1, 3) => 7,
                                _ => unreachable!(),
                            };
                            v |= 1 << dot_idx;
                        }
                    }
                }
                if v == 0 {
                    line.push(' ');
                } else {
                    line.push(std::char::from_u32(0x2800 + v).unwrap_or(' '));
                }
            }
            lines.push(line);
        }
        lines
    };

    let top_lines = draw_view(&|p| p.x, &|p| p.z, min_x, max_x, min_z, max_z);
    let front_lines = draw_view(&|p| p.x, &|p| p.y, min_x, max_x, min_y, max_y);

    println!("\nPreview:");
    println!("{:<60} | Front View (X-Y)", "Top View (X-Z)");
    println!("{:-<60}-+-{:-<60}", "", "");
    for i in 0..height {
        println!("{} | {}", top_lines[i], front_lines[i]);
    }
    println!();
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

    let processor = MeshProcessor::from_file(&args.input, &transform)?;
    let mut particles = processor.voxelize(
        args.resolution,
        args.surface_only,
        args.narrow_band,
        args.phase_sphere,
    )?;

    if args.disintegrate.is_some() || args.wave.is_some() {
        println!("Applying procedural modifiers...");

        let mut seed = 123456789u32;
        let mut rng = || -> f64 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f64) / (u32::MAX as f64)
        };

        #[allow(clippy::collapsible_if)]
        particles.retain_mut(|p| {
            let mut keep = true;

            if let Some(prob) = args.disintegrate {
                if rng() < prob {
                    keep = false;
                }
            }

            if keep {
                if let Some(wave) = args.wave {
                    let amplitude = wave[0] as f32;
                    let frequency = wave[1] as f32;
                    p.y += (p.x * frequency).sin() * amplitude;
                }
            }

            keep
        });
    }

    println!("Generated {} particles.", particles.len());

    if args.preview {
        print_preview(&particles, args.resolution);
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
