cat src/lib.rs | awk '
/let mut hit_xs = Vec::new();/ {
    print "                    let mut hit_xs = Vec::with_capacity(8);"
    next
}
{ print }
' > src/lib.rs.tmp
mv src/lib.rs.tmp src/lib.rs
