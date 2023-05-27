package main

import "core:fmt"
import "core:math/rand"

// Input train set data
TWICE_INPUT :: [][2]f64 {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
}

// This is a example formula of some nn with two parameters
// y = x * w

// For example in ChatGPT a 1 000 000 000 000 parameters

twice_cost ::proc(w: f64, bias: f64,  debug := false) -> f64 {
    input := TWICE_INPUT
    result: f64 = 0.0
    for i := 0; i < len(input); i += 1 {
        expected := input[i][1]
        x := input[i][0]

        // Iterate over all possible parameters
        actual := x * w + bias

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

twice :: proc() {
    r := rand.create(123)
    w := rand.float64_range(1, 10, &r)
    b := rand.float64_range(1, 5, &r)

    // The tweak value to parameter
    eps := 1e-3
    // Because of twice_cost_direction it`s too big, in NN often uses learning rate to minimize shifting
    rate := 1e-3

    for i := 0; i < 100; i += 1 {
        before_training := twice_cost(w, b)
        // The direction of the twice_costs function
        // https://ru.wikipedia.org/wiki/%D0%9F%D1%80%D0%BE%D0%B8%D0%B7%D0%B2%D0%BE%D0%B4%D0%BD%D0%B0%D1%8F_%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%B8
        // From the formula, the (delta)X is the eps, X0 - our parameter, and the f() - twice_cost function
        twice_cost_direction_eps := (twice_cost(w + eps, b) - before_training) / eps
        twice_cost_direction_bias := (twice_cost(w, b + eps) - before_training) / eps

        fmt.printf("%v --------------\n", i)
        fmt.printf("before train\n--------------\n\tquality: %v\n", before_training);

        // Actual model training
        w -= rate * twice_cost_direction_eps
        b -= rate * twice_cost_direction_bias

        fmt.println("--------------")
        fmt.printf("after train\n-------------- \n\tquality: %v, \t w=%v \t b=%v \n", twice_cost(w, b), w, b);
        fmt.println("--------------\n")
    }

    fmt.println("--------------")
    fmt.printf("%v", w);
}