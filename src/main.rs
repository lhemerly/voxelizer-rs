use clap::Parser;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use voxelizer_rs::{MeshProcessor, ParticleHeader};

#[derive(Parser, Debug)]
#[command(author, version, about = "Voxelizer")]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,

    #[arg(short, long, default_value_t = 0.5)]
    resolution: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Input: {}", args.input);
    println!("Resolution: {} mm", args.resolution);

    let processor = MeshProcessor::from_file(&args.input)?;
    let particles = processor.voxelize(args.resolution);

    println!("Generated {} particles.", particles.len());

    let file = File::create(PathBuf::from(&args.output))?;
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
