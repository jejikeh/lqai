package main

import "core:fmt"
import "core:math/rand"

matrix_fill_random :: proc(m: ^[$X][$Y]$T, min, max: T, random := false) {
    r := rand.create(123)
    pr := &r

    if random {
        pr = nil
    }

    for x in 0..<X {
        for y in 0..<Y {
            m[x][y] = rand.float64_range(min, max, pr)
        }
    }
}

matrix_print :: proc(m: [$X][$Y]$T, name := "_") {
    fmt.printf("%v = [\n", name)
    for x in 0..<X {
        for y in 0..<Y {
            fmt.printf("\t%v ", m[x][y])
        }

        fmt.print("\n")
    }
    fmt.println("]")
}

matrix_mul :: proc(a: [$M][$N]$T, b: [N][$P]T) -> (c: [M][P]T) {
    for i in 0..<M {
        for j in 0..<P {
            for k in 0..<N {
                c[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    return
}

matrix_sum :: proc(a: [$M][$N]$T, b: [M][N]T) -> (c: [M][N]T) {
    for i in 0..<M {
            for k in 0..<N {
                c[i][k] += a[i][k] + b[i][k]
        }
    }

    return
}

matrix_sigmoid :: proc(a: ^[$M][$N]$T) {
    for i in 0..<M {
            for k in 0..<N {
                a[i][k] = sigmoid(a[i][k])
        }
    }
}