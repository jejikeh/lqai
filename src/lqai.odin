package main

import "core:math"

sigmoid :: proc(x: f64) -> f64 {
    return 1.0 / (1.0 + math.exp_f64(x))
}