use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use voxelizer_rs::{MeshProcessor, ParticleHeader};

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
}

fn validate_resolution(s: &str) -> Result<f64, String> {
    let val: f64 = s.parse().map_err(|_| format!("`{s}` isn't a number"))?;
    if val > 1e-6 {
        Ok(val)
    } else {
        Err(format!(
            "Resolution must be greater than 1e-6. Provided: {s}"
        ))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Input: {}", args.input);
    println!("Resolution: {} mm", args.resolution);

    let processor = MeshProcessor::from_file(&args.input)?;
    let particles = processor.voxelize(args.resolution, args.surface_only)?;

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
        _ => {
            // Default to BIN
            let header = ParticleHeader {
                version: 1,
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
