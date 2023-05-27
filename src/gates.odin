package main

import "core:fmt"
import "core:math/rand"
import "core:math"


// GATES_Input train set data
GATES_INPUT :: [][3]f64 {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
}

// This is a example formula of some nn with two parameters
// y = x1 * w1 + x2 * w2

// For example in ChatGPT a 1 000 000 000 000 parameters

sigmoid :: proc(x: f64) -> f64 {
    return 1.0 / (1.0 + math.exp_f64(x))
}

gates_cost ::proc(w1: f64, w2: f64, bias: f64,  debug := false) -> f64 {
    input := GATES_INPUT
    result := 0.0
    for i := 0; i < len(input); i += 1 {
        expected := input[i][2]
        // Now we have 2 inputs
        x1 := input[i][0]
        x2 := input[i][1]

        // Iterate over all possible parameters
        actual := sigmoid((x1 * w1) + (x2 * w2) + bias)

        // To get the proper parameter, we need to find the difference between expected and actual.
        distance := actual - expected
        result += distance * distance

        if debug {
            fmt.printf("actual: %v,\t expected: %v \n", actual, expected);
        }
    }

    // Measure how bad this model performs
    // The less the value - the better model perfom
    result /= f64(len(input))
    return result
}

train :: proc(w1: ^f64, w2: ^f64, b: ^f64, eps: f64, rate: f64, it: int, debug := false) {
    for i := 0; i < it; i += 1 {
        before_training := gates_cost(w1^, w2^, b^)
        // The direction of the gates_costs function
        // https://ru.wikipedia.org/wiki/%D0%9F%D1%80%D0%BE%D0%B8%D0%B7%D0%B2%D0%BE%D0%B4%D0%BD%D0%B0%D1%8F_%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%B8
        // From the formula, the (delta)X is the eps, X0 - our parameter, and the f() - gates_cost function
        gates_cost_direction_w1 := (gates_cost(w1^ + eps, w2^, b^) - before_training) / eps
        gates_cost_direction_w2 := (gates_cost(w1^, w2^ + eps, b^) - before_training) / eps
        gates_cost_direction_bias := (gates_cost(w1^, w2^, b^ + eps) - before_training) / eps

        if debug {
            fmt.printf("%v --------------\n", i)
            fmt.printf("before train\n--------------\n\tquality: %v\n", before_training);
        }

        // Actual model training
        w1^ -= rate * gates_cost_direction_w1
        w2^ -= rate * gates_cost_direction_w2
        b^ -= rate * gates_cost_direction_bias

        if debug {
            fmt.println("--------------")
            fmt.printf("after train\n-------------- \n\tquality: %v,\n\t \t w1=%v \t w2=%v \t b=%v \n", gates_cost(w1^, w2^, b^), w1^, w2^, b^);
            fmt.println("--------------\n")
        }
    }
}

init_random_weights ::proc(w1: ^f64, w2: ^f64, b: ^f64, random := false) {
    r := rand.create(123)
    pr := &r

    if random {
        pr = nil
    }

    w1^ = rand.float64_range(1, 10, pr)
    w2^ = rand.float64_range(1, 10, pr)
    b^ = rand.float64_range(1, 5, pr)
}

gates :: proc() {
    input := GATES_INPUT
    w1, w2, b := 0., 0., 0.
    init_random_weights(&w1, &w2, &b, true)

    // The tweak value to parameter
    eps := 1e-3
    // Because of gates_cost_direction it`s too big, in NN often uses learning rate to minimize shifting
    rate := 1e-1

    train(&w1, &w2, &b, eps, rate, 1000000, false)

    fmt.println("--------------")
    fmt.printf("w1=%v, w2=%v, c=%v\n", w1, w2, gates_cost(w1, w2, b));
    fmt.println("--------------")

    for y := 0; y < len(input); y += 1 {
        fmt.printf("%v | %v = %v \n", 
            input[y][0], 
            input[y][1], 
            sigmoid((input[y][0] * w1) + (input[y][1] * w2) + b));
    }
}