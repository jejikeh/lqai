package main

import "core:fmt"
// This file contains NN which do XOR operation

// XOR:
// 0 ^ 0 = 0
// 1 ^ 0 = 1
// 0 ^ 1 = 1
// 1 ^ 1 = 0
XOR_INPUT :: [][3]f64 {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
}

// For build XOR is need more than one neuron. 
// But we can emulate same behavior with one neuron and 3 layers
// 1 layer - OR gate
// 2 layer - NAND gate
// 3 layer - AND gate

// So NN will be behave like (x | y) | ~(x & y) expression

Xor :: struct {
    or_w1: f64,
    or_w2: f64,
    or_b: f64,

    nand_w1: f64,
    nand_w2: f64,
    nand_b: f64,

    and_w1: f64,
    and_w2: f64,
    and_b: f64,
}

rand_xor :: proc(allocator := context.allocator) -> ^Xor {
    x := new(Xor, allocator)

    init_random_weights(&x.or_w1, &x.or_w2, &x.or_b, true)
    init_random_weights(&x.nand_w1, &x.nand_w2, &x.nand_b, true)
    init_random_weights(&x.and_w1, &x.and_w2, &x.and_b, true)
    return x
}

print_xor :: proc(m: ^Xor) {
    fmt.printf("----\n%v\n---", m^)
}

forward :: proc(m: ^Xor, x1: f64, x2: f64) -> f64 {
    // a ->
    //       c
    // b ->
    a := sigmoid(m^.or_w1 * x1 + m^.or_w2 * x2 + m^.or_b)
    b := sigmoid(m^.nand_w1 * x1 + m^.nand_w2 * x2 + m^.nand_b)
    return sigmoid(m^.and_w1 * a + m^.and_w2 * b + m^.and_b) // c
}

xor_cost :: proc(m: ^Xor, debug := false) -> f64 {
    input := XOR_INPUT
    result := 0.0
    for i := 0; i < len(input); i += 1 {
        expected := input[i][2]
        // Now we have 2 inputs
        x1 := input[i][0]
        x2 := input[i][1]

        // Iterate over all possible parameters
        actual := forward(m, x1, x2)

        // To get the proper parameter, we need to find the difference between expected and actual.
        distance := actual - expected
        result += distance * distance

        if debug {
            fmt.printf("actual: %v,\t expected: %v \n", actual, expected);
        }
    }

    // Measure how bad this model performs
    // The less the value - the better model perform
    result /= f64(len(input))
    return result
}

// So, since we deal with 9 parameters, we need more convenient way to calculate the difference
finite_difference :: proc(m: ^Xor, eps: f64, allocator := context.allocator) -> ^Xor {
    diff := new(Xor, allocator)
    c := xor_cost(m)

    saved := m^.or_w1;
    m^.or_w1 += eps
    diff.or_w1 = (xor_cost(m) - c) / eps
    m.or_w1 = saved

    saved = m^.or_w2;
    m^.or_w2 += eps
    diff.or_w2 = (xor_cost(m) - c) / eps
    m.or_w2 = saved

    saved = m^.or_b;
    m^.or_b += eps
    diff.or_b = (xor_cost(m) - c) / eps
    m.or_b = saved

    saved = m^.nand_w1;
    m^.nand_w1 += eps
    diff.nand_w1 = (xor_cost(m) - c) / eps
    m.nand_w1 = saved

    saved = m^.nand_w2;
    m^.nand_w2 += eps
    diff.nand_w2 = (xor_cost(m) - c) / eps
    m.nand_w2 = saved

    saved = m^.nand_b;
    m^.nand_b += eps
    diff.nand_b = (xor_cost(m) - c) / eps
    m.nand_b = saved

    saved = m^.and_w1;
    m^.and_w1 += eps
    diff.and_w1 = (xor_cost(m) - c) / eps
    m.and_w1 = saved

    saved = m^.and_w2;
    m^.and_w2 += eps
    diff.and_w2 = (xor_cost(m) - c) / eps
    m.and_w2 = saved

    saved = m^.and_b;
    m^.and_b += eps
    diff.and_b = (xor_cost(m) - c) / eps
    m.and_b = saved

    return diff
}

apply_difference :: proc(x, y: ^Xor, rate: f64) {
    x^.or_w1 -= rate * y^.or_w1
    x^.or_w2 -= rate * y^.or_w2
    x^.or_b -= rate * y^.or_b

    x^.nand_w1 -= rate * y^.nand_w1
    x^.nand_w2 -= rate * y^.nand_w2
    x^.nand_b -= rate * y^.nand_b

    x^.and_w1 -= rate * y^.and_w1
    x^.and_w2 -= rate * y^.and_w2
    x^.and_b -= rate * y^.and_b    
}

xor :: proc() {
    input := XOR_INPUT
    esp := 1e-3
    rate := 1e-1
    x := rand_xor()

    for i := 0; i < 100 * 100 * 100; i += 1 {
        // fmt.println("\n-------------")
        diff := finite_difference(x, esp)
        // fmt.printf("cost before = \t%.6f\n", xor_cost(x))
        apply_difference(x, diff, rate)
        // fmt.printf("cost after = \t%.6f", xor_cost(x))
    }

    for y := 0; y < len(input); y += 1 {
        fmt.printf("%v | %v = %v \n", 
            input[y][0], 
            input[y][1], 
            forward(x, input[y][0], input[y][1]));
    }
}