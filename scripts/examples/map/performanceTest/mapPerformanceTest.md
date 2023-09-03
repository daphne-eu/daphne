<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Performance Test and Comparison of external Map Function

Daphne external map function documentation and results

## Overview

This file explains the testing of the external map function, that can invoke map Kernels of different programming languages. At the end the results are presented

## Background
DaphneDSL (DAPHNE's domain-specific language for linear and relational algebra) supports the second-order function map().map() expects two inputs, a matrix and a user-defined function (UDF) written in DaphneDSL.
The given UDF must have a single scalar argument and return a single scalar value. map() applies the given UDF to each value in the given matrix.

As DaphneDSL is a domain-specific, rather than a general-purpose language, so implementing more complex UDFs may not always be convenient or even possible. Therefore, the task was to explore how to support UDFs written in other languages. As a staring point Python was choosen as programming language.

## Testing
The tests should show the effectiveness and efficiency of the extended `map()`-kernel:

- The utility test showcases the effectiveness by executing expressions in  the python map kernel, which would be difficult (and maybe not possible) in pure DaphneDSL.

- The performance test showcases the detoritation of efficiency of the extended `map()`-kernel in comparison to DaphneDSLs `map()`-kernel. This is fulfilled by comparing the memory consumption and execution time of the python map kernel and DaphneDSLs map kernel for gradually increased power functions (x^2 to x^17) as UDFs and random generated matrices with no sparsity, a seed of 42 and the sizes [5,10,100]. For testing the [mapFunctionsPerformanceTest.py](/scripts/examples/map/performanceTest/mapFunctionsPerformanceTest.py) was created.


Note that testcases for the general functionality and edge cases (Buffer overflow and underflow, Rounding) are incorporated in the C++ kernel tests in the test file [AlternativeMapTest.cpp](/test/runtime/local/kernels/AlternativeMapTest.cpp).

## Results

### Power Tests with different Data types

### General Utility Test

## Known Limitations in the General Approach

### Conditional Function Definitions (scf.if)
As of the current version, the map function does not support function definitions that depend on conditional control flow (via scf.if operations). Attempting to define a function string conditionally will result in a runtime error.