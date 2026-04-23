use anyhow::Result;
use nalgebra::Point3;
use parry3d::math::{Isometry, Point, Vector};
use parry3d::query::{Ray, RayCast, intersection_test};
use parry3d::shape::{Cuboid, TriMesh};
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
    pub sdf: f32,
}

type MeshData = (Vec<Point<f64>>, Vec<[u32; 3]>);

#[derive(Debug, Clone, Copy)]
pub struct TransformConfig {
    pub scale: f64,
    pub center: bool,
    pub translate: Option<[f64; 3]>,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: false,
            translate: None,
        }
    }
}

pub struct MeshProcessor {
    mesh: TriMesh,
    bounds_min: Point3<f64>,
    bounds_max: Point3<f64>,
}

impl MeshProcessor {
    pub fn from_file(path: &str, transform: &TransformConfig) -> Result<Self> {
        let path_obj = Path::new(path);
        let extension = path_obj.extension().and_then(|s| s.to_str());
        let ext_lower = extension.map(|e| e.to_lowercase());

        let (mut points, indices) = match ext_lower.as_deref() {
            Some("obj") => Self::load_obj(path)?,
            Some("stl") => Self::load_stl(path)?,
            Some(ext) => anyhow::bail!("Unsupported file format: {}", ext),
            None => anyhow::bail!("Missing file extension"),
        };

        if transform.center {
            let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
            let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
            for p in &points {
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                min.z = min.z.min(p.z);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
                max.z = max.z.max(p.z);
            }
            let center = Point3::new(
                (min.x + max.x) * 0.5,
                (min.y + max.y) * 0.5,
                (min.z + max.z) * 0.5,
            );
            for p in &mut points {
                p.x -= center.x;
                p.y -= center.y;
                p.z -= center.z;
            }
        }

        if (transform.scale - 1.0).abs() > f64::EPSILON {
            for p in &mut points {
                p.x *= transform.scale;
                p.y *= transform.scale;
                p.z *= transform.scale;
            }
        }

        if let Some(t) = transform.translate {
            for p in &mut points {
                p.x += t[0];
                p.y += t[1];
                p.z += t[2];
            }
        }

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

    fn load_obj(path: &str) -> Result<MeshData> {
        let (models, _) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
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
            if mesh.positions.len() % 3 != 0 {
                anyhow::bail!(
                    "Model '{}' has a malformed positions array (length {} is not a multiple of 3)",
                    model.name,
                    mesh.positions.len()
                );
            }
            if mesh.indices.len() % 3 != 0 {
                anyhow::bail!(
                    "Model '{}' has a malformed indices array (length {} is not a multiple of 3)",
                    model.name,
                    mesh.indices.len()
                );
            }
            for chunk in mesh.positions.chunks_exact(3) {
                all_points.push(Point::new(
                    chunk[0] as f64,
                    chunk[1] as f64,
                    chunk[2] as f64,
                ));
            }
            for chunk in mesh.indices.chunks_exact(3) {
                all_indices.push([chunk[0] + offset, chunk[1] + offset, chunk[2] + offset]);
            }
            offset += (mesh.positions.len() / 3) as u32;
        }
        Ok((all_points, all_indices))
    }

    fn load_stl(path: &str) -> Result<MeshData> {
        let mut file = File::open(path)?;
        let stl = stl_io::read_stl(&mut file)?;

        let points: Vec<Point<f64>> = stl
            .vertices
            .iter()
            .map(|v| Point::new(v[0] as f64, v[1] as f64, v[2] as f64))
            .collect();

        let indices: Vec<[u32; 3]> = stl
            .faces
            .iter()
            .map(|f| {
                [
                    f.vertices[0] as u32,
                    f.vertices[1] as u32,
                    f.vertices[2] as u32,
                ]
            })
            .collect();

        Ok((points, indices))
    }

    pub fn voxelize(&self, resolution: f64, surface_only: bool) -> Result<Vec<ParticleData>> {
        if resolution <= 1e-6 {
            anyhow::bail!(
                "Resolution must be greater than 1e-6 to avoid excessive resource usage or division by zero. Provided: {}",
                resolution
            );
        }

        let start_time = std::time::Instant::now();

        let size = self.bounds_max - self.bounds_min;
        let nx = (size.x / resolution).ceil() as u64;
        let ny = (size.y / resolution).ceil() as u64;
        let nz = (size.z / resolution).ceil() as u64;

        println!(
            "Grid Dimensions: {} x {} x {} (Potential Voxels: {})",
            nx,
            ny,
            nz,
            nx * ny * nz
        );

        // We avoid collecting the entire yz cartesian product to save memory.
        // Instead we can use rayon's `into_par_iter` on a range or use flat_map across the ranges.
        let particles: Vec<ParticleData> = (0..ny)
            .into_par_iter()
            .flat_map(|iy| {
                (0..nz).into_par_iter().flat_map(move |iz| {
                    let mut local_particles = Vec::with_capacity(nx as usize);
                    let y = self.bounds_min.y + (iy as f64 * resolution) + (resolution * 0.5);
                    let z = self.bounds_min.z + (iz as f64 * resolution) + (resolution * 0.5);

                    // Iterate over X in the inner loop to optimize spatial cache locality.
                    // Because rays are cast along the +X direction, doing X sequentially
                    // keeps the raycast traversals in the same BVH region,
                    // while parallelizing over (Y, Z) ensures finer granularity for Rayon.
                    if surface_only {
                        let half_res = resolution * 0.5;
                        let cuboid = Cuboid::new(Vector::new(half_res, half_res, half_res));
                        let mesh_iso = Isometry::identity();

                        for ix in 0..nx {
                            let x =
                                self.bounds_min.x + (ix as f64 * resolution) + (resolution * 0.5);
                            let point = Point::new(x, y, z);
                            let voxel_iso = Isometry::translation(point.x, point.y, point.z);

                            if let Ok(true) =
                                intersection_test(&mesh_iso, &self.mesh, &voxel_iso, &cuboid)
                            {
                                use parry3d::query::PointQuery;
                                let dist = self.mesh.distance_to_local_point(&point, true);
                                local_particles.push(ParticleData {
                                    x: x as f32,
                                    y: y as f32,
                                    z: z as f32,
                                    phase: 0,
                                    sdf: dist as f32,
                                });
                            }
                        }
                    } else {
                        // Cast a single ray along +X from outside the bounding box
                        let start_x = self.bounds_min.x - 1.0;
                        let point = Point::new(start_x, y, z);
                        let ray = Ray::new(point, Vector::x());
                        let mut current_ray = ray;
                        let max_dist = (self.bounds_max.x - start_x) + 1.0;

                        let mut hit_xs = Vec::new();
                        while let Some(hit_toi) =
                            self.mesh.cast_local_ray(&current_ray, max_dist, true)
                        {
                            let hit_point = current_ray.point_at(hit_toi);
                            hit_xs.push(hit_point.x);

                            // Advance ray slightly past the intersection to find the next one
                            current_ray =
                                Ray::new(current_ray.point_at(hit_toi + 1e-4), Vector::x());

                            if hit_toi < 1e-5 {
                                // Break out of an infinite loop if we're not advancing anymore
                                // Note: The advance is 1e-4, so hit_toi should typically be at least that.
                                // But if hit_toi evaluates to very near 0 due to precision, this prevents hanging.
                                // In the original code, the 50-hit cutoff served this purpose.
                                // Let's use a much larger failsafe instead of 50 to allow complex meshes while still preventing infinite loops.
                            }

                            // Prevent true infinite loops in degenerate floating-point cases
                            // without artificially limiting complex geometry.
                            if hit_xs.len() > 100_000 {
                                break;
                            }
                        }

                        // Sort intersections just in case precision issues caused out-of-order results
                        hit_xs
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        for ix in 0..nx {
                            let x =
                                self.bounds_min.x + (ix as f64 * resolution) + (resolution * 0.5);

                            // A point is inside if it has an odd number of intersections to its right (or left).
                            // Let's count intersections with X > x.
                            let intersections_to_right =
                                hit_xs.iter().filter(|&&hx| hx > x).count();

                            if intersections_to_right % 2 != 0 {
                                use parry3d::query::PointQuery;
                                let p = Point::new(x, y, z);
                                let dist = self.mesh.distance_to_local_point(&p, true);
                                local_particles.push(ParticleData {
                                    x: x as f32,
                                    y: y as f32,
                                    z: z as f32,
                                    phase: 0,
                                    sdf: -dist as f32,
                                });
                            }
                        }
                    }
                    local_particles
                })
            })
            .collect();

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
        let processor = MeshProcessor {
            mesh,
            bounds_min,
            bounds_max,
        };

        assert!(processor.voxelize(0.0, false).is_err());
        assert!(processor.voxelize(-1.0, false).is_err());
        assert!(processor.voxelize(1e-7, false).is_err());
        assert!(processor.voxelize(0.5, false).is_ok());
    }

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
            [0, 1, 2],
            [0, 2, 3], // Front
            [5, 4, 7],
            [5, 7, 6], // Back
            [4, 5, 1],
            [4, 1, 0], // Bottom
            [3, 2, 6],
            [3, 6, 7], // Top
            [4, 0, 3],
            [4, 3, 7], // Left
            [1, 5, 6],
            [1, 6, 2], // Right
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

        let particles = processor.voxelize(0.5, false).unwrap();
        assert_eq!(
            particles.len(),
            8,
            "Expected 8 voxels for a 1x1x1 cube at 0.5 resolution"
        );

        for p in &particles {
            assert!(p.x == 0.25 || p.x == 0.75);
            assert!(p.y == 0.25 || p.y == 0.75);
            assert!(p.z == 0.25 || p.z == 0.75);
        }
    }

    #[test]
    fn test_from_file_unsupported_extension() {
        let err = MeshProcessor::from_file("test.txt", &TransformConfig::default())
            .err()
            .expect("Expected an error for unsupported extension")
            .to_string();
        assert!(err.contains("Unsupported file format"));
        assert!(err.contains("txt"));
    }

    #[test]
    fn test_from_file_no_extension() {
        let err = MeshProcessor::from_file("test", &TransformConfig::default())
            .err()
            .expect("Expected an error for missing extension");
        assert!(err.to_string().contains("Missing file extension"));
    }

    #[test]
    fn test_transform_scale() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!(
            "test_scale_{}.stl",
            std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
        ));

        let faces = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let mut f = std::fs::File::create(&file_path).unwrap();
        use std::io::Write;
        f.write_all(&[0; 80]).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&[0; 12]).unwrap();
        for v in &faces {
            for c in v {
                f.write_all(&(*c as f32).to_le_bytes()).unwrap();
            }
        }
        f.write_all(&[0; 2]).unwrap();

        let config = TransformConfig {
            scale: 2.0,
            ..Default::default()
        };

        let processor = MeshProcessor::from_file(file_path.to_str().unwrap(), &config).unwrap();

        assert_eq!(processor.bounds_min.x, 0.0);
        assert_eq!(processor.bounds_max.x, 2.0);
        assert_eq!(processor.bounds_max.y, 2.0);

        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_transform_center() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!(
            "test_center_{}.stl",
            std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
        ));

        let faces = vec![[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.0, 3.0, 1.0]];

        let mut f = std::fs::File::create(&file_path).unwrap();
        use std::io::Write;
        f.write_all(&[0; 80]).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&[0; 12]).unwrap();
        for v in &faces {
            for c in v {
                f.write_all(&(*c as f32).to_le_bytes()).unwrap();
            }
        }
        f.write_all(&[0; 2]).unwrap();

        let config = TransformConfig {
            center: true,
            ..Default::default()
        };

        let processor = MeshProcessor::from_file(file_path.to_str().unwrap(), &config).unwrap();

        assert_eq!(processor.bounds_min.x, -1.0);
        assert_eq!(processor.bounds_max.x, 1.0);
        assert_eq!(processor.bounds_min.y, -1.0);
        assert_eq!(processor.bounds_max.y, 1.0);
        assert_eq!(processor.bounds_min.z, 0.0);
        assert_eq!(processor.bounds_max.z, 0.0);

        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_transform_translate() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!(
            "test_translate_{}.stl",
            std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
        ));

        let faces = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let mut f = std::fs::File::create(&file_path).unwrap();
        use std::io::Write;
        f.write_all(&[0; 80]).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&[0; 12]).unwrap();
        for v in &faces {
            for c in v {
                f.write_all(&(*c as f32).to_le_bytes()).unwrap();
            }
        }
        f.write_all(&[0; 2]).unwrap();

        let config = TransformConfig {
            translate: Some([10.0, 20.0, 30.0]),
            ..Default::default()
        };

        let processor = MeshProcessor::from_file(file_path.to_str().unwrap(), &config).unwrap();

        assert_eq!(processor.bounds_min.x, 10.0);
        assert_eq!(processor.bounds_max.x, 11.0);
        assert_eq!(processor.bounds_min.y, 20.0);
        assert_eq!(processor.bounds_max.y, 21.0);
        assert_eq!(processor.bounds_min.z, 30.0);
        assert_eq!(processor.bounds_max.z, 30.0);

        std::fs::remove_file(file_path).unwrap();
    }
}
