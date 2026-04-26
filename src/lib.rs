use anyhow::Result;
use nalgebra::Point3;
use parry3d::math::{Isometry, Point, Vector};
use parry3d::query::{PointQuery, Ray, RayCast, intersection_test};
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
    pub sdf: f32,
    pub phase: u32,
    pub label_id: u32,
    pub fiber_x: f32,
    pub fiber_y: f32,
}

type MeshData = (Vec<Point<f64>>, Vec<[u32; 3]>);

#[derive(Debug, Clone, Copy)]
pub struct VoxelizeOptions {
    pub resolution: f64,
    pub surface_only: bool,
    pub narrow_band: Option<f64>,
    pub phase_sphere: Option<[f64; 4]>,
}

#[derive(Debug, Clone, Copy)]
pub struct TransformConfig {
    pub scale: f64,
    pub center: bool,
    pub translate: Option<[f64; 3]>,
    pub rotate: Option<[f64; 3]>,  // x, y, z in degrees
    pub crop: Option<[f64; 6]>,    // min_x, min_y, min_z, max_x, max_y, max_z
    pub vertex_noise: Option<f64>, // random displacement amplitude
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: false,
            translate: None,
            rotate: None,
            crop: None,
            vertex_noise: None,
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

        if let Some(r) = transform.rotate {
            let rx = r[0].to_radians();
            let ry = r[1].to_radians();
            let rz = r[2].to_radians();

            let rot_x = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::x_axis(), rx);
            let rot_y = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), ry);
            let rot_z = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), rz);

            let rotation = rot_z * rot_y * rot_x;

            for p in &mut points {
                *p = rotation * *p;
            }
        }

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

        #[allow(clippy::collapsible_if)]
        if let Some(amp) = transform.vertex_noise {
            if amp > 0.0 {
                // Simple, fast pseudo-random number generator for noise
                let mut seed = 123456789u32;
                for p in &mut points {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let rx = (seed as f64 / u32::MAX as f64) * 2.0 - 1.0;
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let ry = (seed as f64 / u32::MAX as f64) * 2.0 - 1.0;
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let rz = (seed as f64 / u32::MAX as f64) * 2.0 - 1.0;

                    p.x += rx * amp;
                    p.y += ry * amp;
                    p.z += rz * amp;
                }
            }
        }

        let mesh = TriMesh::new(points, indices);
        let aabb = mesh.local_aabb();

        let mut bounds_min = aabb.mins;
        let mut bounds_max = aabb.maxs;

        if let Some(crop) = transform.crop {
            bounds_min.x = bounds_min.x.max(crop[0]);
            bounds_min.y = bounds_min.y.max(crop[1]);
            bounds_min.z = bounds_min.z.max(crop[2]);
            bounds_max.x = bounds_max.x.min(crop[3]);
            bounds_max.y = bounds_max.y.min(crop[4]);
            bounds_max.z = bounds_max.z.min(crop[5]);
        }

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

    pub fn voxelize(&self, options: &VoxelizeOptions) -> Result<Vec<ParticleData>> {
        if !options.resolution.is_finite() || options.resolution <= 1e-6 {
            anyhow::bail!(
                "Resolution must be a finite number greater than 1e-6 to avoid excessive resource usage or division by zero. Provided: {}",
                options.resolution
            );
        }

        if options
            .narrow_band
            .is_some_and(|band| !band.is_finite() || band < 0.0)
        {
            anyhow::bail!(
                "Narrow band must be a finite non-negative number. Provided: {}",
                options.narrow_band.unwrap()
            );
        }

        let resolution = options.resolution;
        let surface_only = options.surface_only;
        let narrow_band = options.narrow_band;
        let phase_sphere = options.phase_sphere;

        let start_time = std::time::Instant::now();

        let mut bounds_min = self.bounds_min;
        let mut bounds_max = self.bounds_max;

        if let Some(band) = narrow_band {
            // Expand the evaluation bounds by the narrow_band to ensure we capture
            // exterior voxels in the required shell.
            bounds_min.x -= band;
            bounds_min.y -= band;
            bounds_min.z -= band;
            bounds_max.x += band;
            bounds_max.y += band;
            bounds_max.z += band;
        }

        let size = bounds_max - bounds_min;
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
                    let y = bounds_min.y + (iy as f64 * resolution) + (resolution * 0.5);
                    let z = bounds_min.z + (iz as f64 * resolution) + (resolution * 0.5);

                    // Extract raycasting to be available for both modes so SDF sign is consistent.
                    let start_x = self.bounds_min.x - 1.0;
                    let ray_point = Point::new(start_x, y, z);
                    let ray = Ray::new(ray_point, Vector::x());
                    let mut current_ray = ray;
                    let max_dist = (bounds_max.x.max(self.bounds_max.x) - start_x) + 1.0;

                    let mut hit_xs = Vec::new();
                    while let Some(hit_toi) = self.mesh.cast_local_ray(&current_ray, max_dist, true)
                    {
                        let hit_point = current_ray.point_at(hit_toi);
                        hit_xs.push(hit_point.x);

                        current_ray = Ray::new(current_ray.point_at(hit_toi + 1e-4), Vector::x());

                        if hit_xs.len() > 100_000 {
                            break;
                        }
                    }

                    // Sort intersections just in case precision issues caused out-of-order results
                    hit_xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let add_particle =
                        |x: f64,
                         y: f64,
                         z: f64,
                         sdf: f32,
                         local_particles: &mut Vec<ParticleData>| {
                            let mut phase = 0;
                            if let Some(sphere) = phase_sphere {
                                let dx = x - sphere[0];
                                let dy = y - sphere[1];
                                let dz = z - sphere[2];
                                let r2 = sphere[3] * sphere[3];
                                if dx * dx + dy * dy + dz * dz <= r2 {
                                    phase = 1;
                                }
                            }
                            local_particles.push(ParticleData {
                                x: x as f32,
                                y: y as f32,
                                z: z as f32,
                                sdf,
                                phase,
                                label_id: 0,
                                fiber_x: 0.0,
                                fiber_y: 0.0,
                            });
                        };

                    // Iterate over X in the inner loop to optimize spatial cache locality.
                    // Because rays are cast along the +X direction, doing X sequentially
                    // keeps the raycast traversals in the same BVH region,
                    // while parallelizing over (Y, Z) ensures finer granularity for Rayon.
                    if surface_only {
                        let half_res = resolution * 0.5;
                        let cuboid = Cuboid::new(Vector::new(half_res, half_res, half_res));
                        let mesh_iso = Isometry::identity();

                        for ix in 0..nx {
                            let x = bounds_min.x + (ix as f64 * resolution) + (resolution * 0.5);
                            let point = Point::new(x, y, z);
                            let voxel_iso = Isometry::translation(point.x, point.y, point.z);

                            if let Ok(true) =
                                intersection_test(&mesh_iso, &self.mesh, &voxel_iso, &cuboid)
                            {
                                let intersections_to_right =
                                    hit_xs.len() - hit_xs.partition_point(|&hx| hx <= x);
                                let is_inside = intersections_to_right % 2 != 0;

                                let distance =
                                    self.mesh.distance_to_local_point(&point, false) as f32;
                                let sdf = if is_inside { -distance } else { distance };

                                // Surface voxels inherently intersect the surface, so they should always be kept
                                // if we're not using narrow_band. If narrow_band is used, we check the distance.
                                let keep = if let Some(band) = narrow_band {
                                    sdf.abs() <= band as f32
                                } else {
                                    true
                                };

                                if keep {
                                    add_particle(x, y, z, sdf, &mut local_particles);
                                }
                            }
                        }
                    } else {
                        for ix in 0..nx {
                            let x = bounds_min.x + (ix as f64 * resolution) + (resolution * 0.5);
                            let point_3d = Point::new(x, y, z);

                            // A point is inside if it has an odd number of intersections to its right (or left).
                            // hit_xs is sorted, so we can use partition_point for O(log N) lookup.
                            let intersections_to_right =
                                hit_xs.len() - hit_xs.partition_point(|&hx| hx <= x);

                            let is_inside = intersections_to_right % 2 != 0;

                            let distance =
                                self.mesh.distance_to_local_point(&point_3d, false) as f32;
                            let sdf = if is_inside { -distance } else { distance };

                            let keep = if let Some(band) = narrow_band {
                                sdf.abs() <= band as f32
                            } else {
                                sdf <= 0.0
                            };

                            if keep {
                                add_particle(x, y, z, sdf, &mut local_particles);
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

        let check_err = |res: f64| {
            let options = VoxelizeOptions {
                resolution: res,
                surface_only: false,
                narrow_band: None,
                phase_sphere: None,
            };
            let err = processor.voxelize(&options).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "Resolution must be a finite number greater than 1e-6 to avoid excessive resource usage or division by zero. Provided: {}",
                    res
                )
            );
        };

        check_err(0.0);
        check_err(-1.0);
        check_err(1e-7);
        check_err(f64::NAN);
        check_err(f64::INFINITY);

        let valid_options = VoxelizeOptions {
            resolution: 0.5,
            surface_only: false,
            narrow_band: None,
            phase_sphere: None,
        };
        assert!(processor.voxelize(&valid_options).is_ok());
    }

    #[test]
    fn test_voxelize_invalid_narrow_band() {
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

        let assert_narrow_band_error = |band: f64| {
            let options = VoxelizeOptions {
                resolution: 0.5,
                surface_only: false,
                narrow_band: Some(band),
                phase_sphere: None,
            };
            let err = processor.voxelize(&options).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "Narrow band must be a finite non-negative number. Provided: {}",
                    band
                )
            );
        };

        assert_narrow_band_error(-1.0);
        assert_narrow_band_error(f64::NAN);
        assert_narrow_band_error(f64::INFINITY);
        assert_narrow_band_error(f64::NEG_INFINITY);

        let valid_options_1 = VoxelizeOptions {
            resolution: 0.5,
            surface_only: false,
            narrow_band: Some(0.0),
            phase_sphere: None,
        };
        assert!(processor.voxelize(&valid_options_1).is_ok());

        let valid_options_2 = VoxelizeOptions {
            resolution: 0.5,
            surface_only: false,
            narrow_band: Some(2.0),
            phase_sphere: None,
        };
        assert!(processor.voxelize(&valid_options_2).is_ok());
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

        let options = VoxelizeOptions {
            resolution: 0.5,
            surface_only: false,
            narrow_band: None,
            phase_sphere: None,
        };
        let particles = processor.voxelize(&options).unwrap();
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

    #[test]
    fn test_transform_crop() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!(
            "test_crop_{}.stl",
            std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
        ));

        let faces = vec![
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [0.0, 10.0, 10.0]],
        ];

        let mut f = std::fs::File::create(&file_path).unwrap();
        use std::io::Write;
        f.write_all(&[0; 80]).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        for v in &faces {
            f.write_all(&[0; 12]).unwrap();
            for pt in v {
                for c in pt {
                    f.write_all(&(*c as f32).to_le_bytes()).unwrap();
                }
            }
            f.write_all(&[0; 2]).unwrap();
        }

        let config = TransformConfig {
            crop: Some([2.0, 2.0, 2.0, 8.0, 8.0, 8.0]),
            ..Default::default()
        };

        let processor = MeshProcessor::from_file(file_path.to_str().unwrap(), &config).unwrap();

        assert_eq!(processor.bounds_min.x, 2.0);
        assert_eq!(processor.bounds_min.y, 2.0);
        assert_eq!(processor.bounds_min.z, 2.0);
        assert_eq!(processor.bounds_max.x, 8.0);
        assert_eq!(processor.bounds_max.y, 8.0);
        assert_eq!(processor.bounds_max.z, 8.0);

        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_transform_rotate() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(format!(
            "test_rotate_{}.stl",
            std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
        ));

        let faces = vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

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
            rotate: Some([0.0, 0.0, 90.0]),
            ..Default::default()
        };

        let processor = MeshProcessor::from_file(file_path.to_str().unwrap(), &config).unwrap();

        // The point (1,0,0) rotated 90 degrees around Z should become (0,1,0)
        assert!((processor.bounds_min.x - 0.0).abs() < 1e-5);
        assert!((processor.bounds_max.y - 1.0).abs() < 1e-5);

        std::fs::remove_file(file_path).unwrap();
    }
}
