use anyhow::Result;
use nalgebra::Point3;
use parry3d::math::{Point, Vector};
use parry3d::query::{Ray, RayCast};
use parry3d::shape::TriMesh;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct ParticleHeader {
    pub version: u32,
    pub particle_count: u64,
    pub resolution: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ParticleData {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub phase: u32,
}

pub struct MeshProcessor {
    mesh: TriMesh,
    bounds_min: Point3<f64>,
    bounds_max: Point3<f64>,
}

impl MeshProcessor {
    pub fn from_file(path: &str) -> Result<Self> {
        let path_obj = Path::new(path);
        let extension = path_obj.extension().and_then(|s| s.to_str());
        let ext_lower = extension.map(|e| e.to_lowercase());

        let (points, indices) = match ext_lower.as_deref() {
            Some("obj") => Self::load_obj(path)?,
            Some("stl") => Self::load_stl(path)?,
            Some(ext) => anyhow::bail!("Unsupported file format: {}", ext),
            None => anyhow::bail!("Missing file extension"),
        };

        let mesh = TriMesh::new(points, indices);
        let aabb = mesh.local_aabb();
        
        let bounds_min = aabb.mins;
        let bounds_max = aabb.maxs;

        Ok(Self {
            mesh,
            bounds_min,
            bounds_max,
        })
    }

    fn load_obj(path: &str) -> Result<(Vec<Point<f64>>, Vec<[u32; 3]>)> {
        let (models, _) = tobj::load_obj(
            path,
            &tobj::LoadOptions { triangulate: true, ..Default::default() }
        )?;
        
        // Pre-calculate capacities for all vertex positions and triangle faces to avoid
        // multiple reallocations during vector growth, especially for large OBJ files.
        let (total_points, total_faces) = models.iter().fold((0, 0), |acc, model| {
            (
                acc.0 + model.mesh.positions.len() / 3,
                acc.1 + model.mesh.indices.len() / 3,
            )
        });

        let mut all_points = Vec::with_capacity(total_points);
        let mut all_indices = Vec::with_capacity(total_faces);
        let mut offset = 0;

        for model in models {
            let mesh = model.mesh;
            let pos_chunks = mesh.positions.chunks_exact(3);
            if !pos_chunks.remainder().is_empty() {
                anyhow::bail!(
                    "OBJ positions length ({}) is not divisible by 3; malformed mesh data",
                    mesh.positions.len()
                );
            }
            for chunk in pos_chunks {
                all_points.push(Point::new(
                    chunk[0] as f64,
                    chunk[1] as f64,
                    chunk[2] as f64,
                ));
            }
            let idx_chunks = mesh.indices.chunks_exact(3);
            if !idx_chunks.remainder().is_empty() {
                anyhow::bail!(
                    "OBJ indices length ({}) is not divisible by 3; malformed mesh data",
                    mesh.indices.len()
                );
            }
            for chunk in idx_chunks {
                all_indices.push([
                    chunk[0] + offset,
                    chunk[1] + offset,
                    chunk[2] + offset,
                ]);
            }
            offset += (mesh.positions.len() / 3) as u32;
        }
        Ok((all_points, all_indices))
    }

    fn load_stl(path: &str) -> Result<(Vec<Point<f64>>, Vec<[u32; 3]>)> {
        let mut file = File::open(path)?;
        let stl = stl_io::read_stl(&mut file)?;
        
        let points: Vec<Point<f64>> = stl.vertices.iter()
            .map(|v| Point::new(v[0] as f64, v[1] as f64, v[2] as f64))
            .collect();

        let indices: Vec<[u32; 3]> = stl.faces.iter()
            .map(|f| [
                f.vertices[0] as u32, 
                f.vertices[1] as u32, 
                f.vertices[2] as u32
            ])
            .collect();
            
        Ok((points, indices))
    }

    pub fn voxelize(&self, resolution: f64) -> Result<Vec<ParticleData>> {
        if resolution <= 1e-6 {
            anyhow::bail!("Resolution must be greater than 1e-6 to avoid excessive resource usage or division by zero. Provided: {}", resolution);
        }

        let start_time = std::time::Instant::now();
        
        let size = self.bounds_max - self.bounds_min;
        let nx = (size.x / resolution).ceil() as u64;
        let ny = (size.y / resolution).ceil() as u64;
        let nz = (size.z / resolution).ceil() as u64;
        
        println!("Grid Dimensions: {} x {} x {} (Potential Voxels: {})", nx, ny, nz, nx*ny*nz);

        // We avoid collecting the entire yz cartesian product to save memory.
        // Instead we can use rayon's `into_par_iter` on a range or use flat_map across the ranges.
        let particles: Vec<ParticleData> = (0..ny).into_par_iter().flat_map(|iy| {
            (0..nz).into_par_iter().flat_map(move |iz| {
                let mut local_particles = Vec::with_capacity(nx as usize);
                let y = self.bounds_min.y + (iy as f64 * resolution) + (resolution * 0.5);
                let z = self.bounds_min.z + (iz as f64 * resolution) + (resolution * 0.5);

            // Iterate over X in the inner loop to optimize spatial cache locality.
            // Because rays are cast along the +X direction, doing X sequentially
            // keeps the raycast traversals in the same BVH region,
            // while parallelizing over (Y, Z) ensures finer granularity for Rayon.
            for ix in 0..nx {
                let x = self.bounds_min.x + (ix as f64 * resolution) + (resolution * 0.5);
                let point = Point::new(x, y, z);

                let ray = Ray::new(point, Vector::x());
                let mut intersections = 0;
                let mut current_ray = ray;
                let max_dist = 1000.0;

                while let Some(hit) = self.mesh.cast_local_ray_and_get_normal(&current_ray, max_dist, true) {
                    intersections += 1;
                    let hit_point = current_ray.point_at(hit.toi + 1e-4);
                    current_ray = Ray::new(hit_point, Vector::x());
                    
                    if intersections > 20 { break; }
                }

                if intersections % 2 != 0 {
                    local_particles.push(ParticleData {
                        x: x as f32,
                        y: y as f32,
                        z: z as f32,
                        phase: 0,
                    });
                }
            }
            local_particles
            })
        }).collect();

        let duration = start_time.elapsed();
        println!("Voxelization complete in {:.2?}s", duration);
        
        Ok(particles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxelize_invalid_resolution() {
        let points = vec![
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
            Point::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![[0, 1, 2]];
        let mesh = TriMesh::new(points, indices);
        let bounds_min = Point3::new(0.0, 0.0, 0.0);
        let bounds_max = Point3::new(1.0, 1.0, 1.0);
        let processor = MeshProcessor { mesh, bounds_min, bounds_max };

        assert!(processor.voxelize(0.0).is_err());
        assert!(processor.voxelize(-1.0).is_err());
        assert!(processor.voxelize(1e-7).is_err());
        assert!(processor.voxelize(1e-5).is_ok());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use parry3d::math::Point;

    #[test]
    fn test_voxelize_cube() {
        let points = vec![
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
            Point::new(1.0, 1.0, 0.0),
            Point::new(0.0, 1.0, 0.0),
            Point::new(0.0, 0.0, 1.0),
            Point::new(1.0, 0.0, 1.0),
            Point::new(1.0, 1.0, 1.0),
            Point::new(0.0, 1.0, 1.0),
        ];

        let indices = vec![
            [0, 1, 2], [0, 2, 3], // Front
            [5, 4, 7], [5, 7, 6], // Back
            [4, 5, 1], [4, 1, 0], // Bottom
            [3, 2, 6], [3, 6, 7], // Top
            [4, 0, 3], [4, 3, 7], // Left
            [1, 5, 6], [1, 6, 2], // Right
        ];

        let mesh = TriMesh::new(points, indices);
        let aabb = mesh.local_aabb();
        let bounds_min = aabb.mins;
        let bounds_max = aabb.maxs;

        let processor = MeshProcessor {
            mesh,
            bounds_min,
            bounds_max,
        };

        let particles = processor.voxelize(0.5);
        assert_eq!(particles.len(), 8, "Expected 8 voxels for a 1x1x1 cube at 0.5 resolution");

        for p in &particles {
            assert!(p.x == 0.25 || p.x == 0.75);
            assert!(p.y == 0.25 || p.y == 0.75);
            assert!(p.z == 0.25 || p.z == 0.75);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_file_unsupported_extension() {
        let err = MeshProcessor::from_file("test.txt")
            .err()
            .expect("Expected an error for unsupported extension")
            .to_string();
        assert!(err.contains("Unsupported file format"));
        assert!(err.contains("txt"));
    }

    #[test]
    fn test_from_file_no_extension() {
        let err = MeshProcessor::from_file("test")
            .err()
            .expect("Expected an error for missing extension");
        assert!(err.to_string().contains("Missing file extension"));
    }
}
