sed -i 's/let distance =/let sdf = if narrow_band.is_none() \&\& !is_inside { f32::INFINITY } else { let distance =/g' src/lib.rs
sed -i 's/self.mesh.distance_to_local_point(&point_3d, false) as f32;/self.mesh.distance_to_local_point(\&point_3d, false) as f32; if is_inside { -distance } else { distance } };/g' src/lib.rs
sed -i 's/let sdf = if is_inside { -distance } else { distance };//g' src/lib.rs
