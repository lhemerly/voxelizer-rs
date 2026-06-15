sed -i 's/let mut hit_xs = Vec::new();/let mut hit_xs = Vec::with_capacity(8);/g' src/lib.rs
