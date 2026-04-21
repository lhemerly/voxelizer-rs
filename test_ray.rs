use parry3d::math::{Point, Vector};
use parry3d::query::{Ray, RayCast};
use parry3d::shape::TriMesh;
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let nx = 100_u64;
    let ny = 100_u64;
    let nz = 100_u64;
    let resolution = 0.01;

    // Simulate Rayon parallel iteration over Z and Y, then inner loop over X vs
    // parallel iteration over Z and X, then inner loop over Y vs
    // parallel iteration over X and Y, then inner loop over Z.

    println!("See if we can optimize Raycast direction and loops.");
}
