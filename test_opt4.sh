cat src/lib.rs | awk '
BEGIN { in_else = 0 }
/\} else \{/ && /for ix in 0..nx/ { in_else = 1 }
/let distance =/ && in_else == 1 {
    if (!done) {
        print "                            let sdf = if narrow_band.is_none() && !is_inside {"
        print "                                f32::INFINITY"
        print "                            } else {"
        print "                                let distance ="
        print "                                    self.mesh.distance_to_local_point(&point_3d, false) as f32;"
        print "                                if is_inside { -distance } else { distance }"
        print "                            };"
        done = 1
        skip = 2
        next
    }
}
skip > 0 { skip--; next }
/let sdf = if is_inside { -distance } else { distance };/ && in_else == 1 { next }
{ print }
' > src/lib.rs.tmp
mv src/lib.rs.tmp src/lib.rs
