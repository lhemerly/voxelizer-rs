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

unsafe impl Send for ParticleData {}
unsafe impl Sync for ParticleData {}

pub struct MeshProcessor {
    mesh: TriMesh,
    bounds_min: Point3<f64>,
    bounds_max: Point3<f64>,
}

impl MeshProcessor {
    pub fn from_file(path: &str) -> Result<Self> {
        let path_obj = Path::new(path);
        let extension = path_obj.extension().and_then(|s| s.to_str()).unwrap_or("");

        let (points, indices) = match extension.to_lowercase().as_str() {
            "obj" => Self::load_obj(path)?,
            "stl" => Self::load_stl(path)?,
            _ => anyhow::bail!("Unsupported file format: {}", extension),
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
        
        let mut all_points = Vec::new();
        let mut all_indices = Vec::new();
        let mut offset = 0;

        for model in models {
            let mesh = model.mesh;
            for i in 0..mesh.positions.len() / 3 {
                all_points.push(Point::new(
                    mesh.positions[i * 3] as f64,
                    mesh.positions[i * 3 + 1] as f64,
                    mesh.positions[i * 3 + 2] as f64,
                ));
            }
            for i in 0..mesh.indices.len() / 3 {
                all_indices.push([
                    mesh.indices[i * 3] + offset,
                    mesh.indices[i * 3 + 1] + offset,
                    mesh.indices[i * 3 + 2] + offset,
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

    pub fn voxelize(&self, resolution: f64) -> Vec<ParticleData> {
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
                let mut local_particles = Vec::new();
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
        
        particles
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::io::Write;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn get_temp_file_path(name: &str) -> String {
        static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut path = env::temp_dir();
        let unique_id = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        path.push(format!("test_mesh_{}_{}_{}", process::id(), unique_id, name));
        path.to_string_lossy().into_owned()
    }

    /// RAII guard that deletes the file at the given path when dropped.
    struct TempFile(String);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn test_from_file_obj() {
        let path = get_temp_file_path("test.obj");
        let _guard = TempFile(path.clone());

        let mut file = File::create(&path).unwrap();
        writeln!(file, "v 0.0 0.0 0.0").unwrap();
        writeln!(file, "v 1.0 0.0 0.0").unwrap();
        writeln!(file, "v 0.0 1.0 0.0").unwrap();
        writeln!(file, "f 1 2 3").unwrap();
        drop(file);

        let processor_result = MeshProcessor::from_file(&path);
        assert!(processor_result.is_ok());
        let processor = processor_result.unwrap();
        assert_eq!(processor.bounds_min.x, 0.0);
        assert_eq!(processor.bounds_max.x, 1.0);
    }

    #[test]
    fn test_from_file_stl() {
        let path = get_temp_file_path("test.stl");
        let _guard = TempFile(path.clone());

        let mut file = File::create(&path).unwrap();
        writeln!(file, "solid test").unwrap();
        writeln!(file, "  facet normal 0.0 0.0 1.0").unwrap();
        writeln!(file, "    outer loop").unwrap();
        writeln!(file, "      vertex 0.0 0.0 0.0").unwrap();
        writeln!(file, "      vertex 1.0 0.0 0.0").unwrap();
        writeln!(file, "      vertex 0.0 1.0 0.0").unwrap();
        writeln!(file, "    endloop").unwrap();
        writeln!(file, "  endfacet").unwrap();
        writeln!(file, "endsolid test").unwrap();
        drop(file);

        let processor_result = MeshProcessor::from_file(&path);
        assert!(processor_result.is_ok());
    }

    #[test]
    fn test_from_file_unsupported_extension() {
        let path = get_temp_file_path("test.txt");
        let _guard = TempFile(path.clone());

        let mut file = File::create(&path).unwrap();
        writeln!(file, "dummy content").unwrap();
        drop(file);

        let result = MeshProcessor::from_file(&path);
        match result {
            Err(e) => assert!(e.to_string().contains("Unsupported file format")),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_from_file_no_extension() {
        let path = get_temp_file_path("testfile_no_ext");
        let _guard = TempFile(path.clone());

        let mut file = File::create(&path).unwrap();
        writeln!(file, "dummy content").unwrap();
        drop(file);

        let result = MeshProcessor::from_file(&path);
        match result {
            Err(e) => assert!(e.to_string().contains("Unsupported file format")),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_from_file_non_existent() {
        let path = get_temp_file_path("does_not_exist.obj");
        // Do not create the file — just verify loading a missing file returns an error.
        let result = MeshProcessor::from_file(&path);
        assert!(result.is_err());
    }
}
