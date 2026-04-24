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
    if val.is_finite() {
        Ok(val)
    } else {
        Err(format!(
            "narrow_band must be a finite number. Provided: {s}"
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
    let particles = processor.voxelize(
        args.resolution,
        args.surface_only,
        args.narrow_band,
        args.phase_sphere,
    )?;

    println!("Generated {} particles.", particles.len());

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
