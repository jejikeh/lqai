package main

import "core:fmt"
import "core:math"

main :: proc() {
    a := [7][10]f64 {}
    matrix_fill_random(&a, 0, 1, true)
    b := [7][10]f64 {}
    matrix_fill_random(&b, 0, 1, true)

    matrix_print(a, "a")
    matrix_print(b, "b")

    c := matrix_sum(a, b)
    matrix_print(c, "a + b")
}