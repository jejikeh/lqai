package main

// This file contains NN which do XOR operation

// XOR:
// 0 ^ 0 = 0
// 1 ^ 0 = 1
// 0 ^ 1 = 1
// 1 ^ 1 = 0

// For build XOR is need more than one neuron. 
// But we can emulate same behaveour with one neuron and 3 layers
// 1 layer - OR gate
// 2 layer - NAND gate
// 3 layer - AND gate

// So NN will be behave like (x | y) | ~(x & y) expression