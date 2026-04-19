use clap::Parser;
use std::fs::File;
use std::io::BufWriter;
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
}

fn validate_resolution(s: &str) -> Result<f64, String> {
    let val: f64 = s.parse().map_err(|_| format!("`{s}` isn't a number"))?;
    if val.is_finite() && val > 1e-6 {
        Ok(val)
    } else {
        Err(format!("Resolution must be a finite number greater than 1e-6. Provided: {s}"))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Input: {}", args.input);
    println!("Resolution: {} mm", args.resolution);

    let processor = MeshProcessor::from_file(&args.input)?;
    let particles = processor.voxelize(args.resolution)?;
    
    println!("Generated {} particles.", particles.len());

    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);

    let header = ParticleHeader {
        version: 1,
        particle_count: particles.len() as u64,
        resolution: args.resolution,
    };

    bincode::serialize_into(&mut writer, &header)?;
    bincode::serialize_into(&mut writer, &particles)?;

    println!("Saved to {}", args.output);
    Ok(())
}
