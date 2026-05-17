git restore src/lib.rs

cat << 'PATCH' > patch.diff
<<<<<<< SEARCH
        let particles: Vec<ParticleData> = (0..ny)
            .into_par_iter()
            .flat_map(|iy| {
                (0..nz).into_par_iter().flat_map(move |iz| {
                    let mut local_particles = Vec::with_capacity(nx as usize);
                    let y = bounds_min.y + (iy as f64 * resolution) + (resolution * 0.5);
                    let z = bounds_min.z + (iz as f64 * resolution) + (resolution * 0.5);

                    // Extract raycasting to be available for both modes so SDF sign is consistent.
=======
        let half_res = resolution * 0.5;
        let base_x = bounds_min.x + half_res;
        let base_y = bounds_min.y + half_res;
        let base_z = bounds_min.z + half_res;

        // We avoid collecting the entire yz cartesian product to save memory.
        // Instead we can use rayon's `into_par_iter` on a range or use flat_map across the ranges.
        let particles: Vec<ParticleData> = (0..ny)
            .into_par_iter()
            .flat_map(|iy| {
                let y = base_y + (iy as f64 * resolution);

                (0..nz).into_par_iter().flat_map(move |iz| {
                    let mut local_particles = Vec::with_capacity(nx as usize);
                    let z = base_z + (iz as f64 * resolution);

                    let mut sphere_info = None;
                    if let Some(sphere) = phase_sphere {
                        let cx = sphere[0];
                        let dy = y - sphere[1];
                        let dz = z - sphere[2];
                        let r2 = sphere[3] * sphere[3];
                        sphere_info = Some((cx, dy * dy + dz * dz, r2));
                    }

                    // Extract raycasting to be available for both modes so SDF sign is consistent.
>>>>>>> REPLACE
<<<<<<< SEARCH
                    if surface_only {
                        let half_res = resolution * 0.5;
                        let cuboid = Cuboid::new(Vector::new(half_res, half_res, half_res));
                        let mesh_iso = Isometry::identity();

                        for ix in 0..nx {
                            let x = bounds_min.x + (ix as f64 * resolution) + (resolution * 0.5);
                            let point = Point::new(x, y, z);
=======
                    if surface_only {
                        let cuboid = Cuboid::new(Vector::new(half_res, half_res, half_res));
                        let mesh_iso = Isometry::identity();

                        for ix in 0..nx {
                            let x = base_x + (ix as f64 * resolution);
                            let point = Point::new(x, y, z);
>>>>>>> REPLACE
<<<<<<< SEARCH
                                if keep {
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
=======
                                if keep {
                                    let mut phase = 0;
                                    if let Some((cx, dy2_plus_dz2, r2)) = sphere_info {
                                        let dx = x - cx;
                                        if dx * dx + dy2_plus_dz2 <= r2 {
                                            phase = 1;
                                        }
                                    }
                                    local_particles.push(ParticleData {
>>>>>>> REPLACE
<<<<<<< SEARCH
                    } else {
                        for ix in 0..nx {
                            let x = bounds_min.x + (ix as f64 * resolution) + (resolution * 0.5);
                            let point_3d = Point::new(x, y, z);

                            // A point is inside if it has an odd number of intersections to its right (or left).
=======
                    } else {
                        for ix in 0..nx {
                            let x = base_x + (ix as f64 * resolution);
                            let point_3d = Point::new(x, y, z);

                            // A point is inside if it has an odd number of intersections to its right (or left).
>>>>>>> REPLACE
<<<<<<< SEARCH
                            if keep {
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
=======
                            if keep {
                                let mut phase = 0;
                                if let Some((cx, dy2_plus_dz2, r2)) = sphere_info {
                                    let dx = x - cx;
                                    if dx * dx + dy2_plus_dz2 <= r2 {
                                        phase = 1;
                                    }
                                }
                                local_particles.push(ParticleData {
>>>>>>> REPLACE
PATCH

python3 -c "
import sys
content = open('src/lib.rs').read()
blocks = open('patch.diff').read().split('<<<<<<< SEARCH\n')[1:]
for block in blocks:
    search, replace = block.split('=======\n')
    replace = replace.split('>>>>>>> REPLACE\n')[0]
    if search not in content:
        print('Could not find:\n', search)
        sys.exit(1)
    content = content.replace(search, replace, 1)
open('src/lib.rs', 'w').write(content)
"
cargo test
cargo bench
