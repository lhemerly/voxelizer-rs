## 2024-06-15 - Fast scanline caching pre-allocation optimization

**Learning:** When performing scanline parity checking against sorted ray intersections, pre-allocating the `hit_xs` vector with `Vec::with_capacity(8)` (where 8 is an empirical average bounds crossing count) inside the nested Rayon `par_iter` loop avoids repeated heap allocations and dynamic resizing overhead in performance-critical code paths.
**Action:** When a loop involves generating a vector that is usually small and predictable, swap `Vec::new()` for `Vec::with_capacity(size)`. This can yield measurable (e.g. 5-7%) performance improvements in hot loops.
