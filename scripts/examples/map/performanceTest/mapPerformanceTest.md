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

## Overview

This file explains the testing of the external map function, that can invoke map Kernels of different programming languages. At the end the results are presented

## Background
DaphneDSL (DAPHNE's domain-specific language for linear and relational algebra) supports the second-order function map(). The map function expects two inputs, a matrix and a user-defined function (UDF) written in DaphneDSL. The given UDF must have a single scalar argument and return a single scalar value. map() applies the given UDF to each value in the given matrix.

As DaphneDSL is a domain-specific, rather than a general-purpose language, so implementing more complex UDFs may not always be convenient or even possible. Therefore, the task was to explore how to support UDFs written in other languages. As a staring point Python was choosen as programming language.

## Testing
The tests should show the effectiveness and efficiency of the extended `map()`-kernel:

- The utility test showcases the effectiveness by executing functions in the python map kernel, which would not possible in pure DaphneDSL.

- The performance test showcases the detoritation of efficiency of the extended `map()`-kernel in comparison to DaphneDSLs `map()`-kernel, a Numpy approach in Python and Daphnes Internal elementwise matrix operations. This is fulfilled by comparing the memory consumption, execution time and CPU load of the different implementations for different arithmetic operations as UDFs and random generated matrices with no sparsity, a seed of 42 and different sizes.

Note that testcases for the general functionality and edge cases (Buffer overflow and underflow, Rounding) are incorporated in the C++ kernel tests in the test file [AlternativeMapTest.cpp](/test/runtime/local/kernels/AlternativeMapTest.cpp).

## Results

## Known Limitations in the General Approach

### Conditional Function Definitions (scf.if)
#### Problem:
As of the current version, the map function does not support function definitions that depend on conditional control flow (via scf.if operations). Attempting to define a function string conditionally will result in a runtime error.

#### Mitigation through proper Lowering Op implementation 
Implement a proper handling for the lowering of Strings for the map function or try to integrate it into the GeneralCallOp infrastructure.

### Safety Concerns of using exec in the Python map kernel
#### Problem:
Using the exec() function to execute Python code can introduce significant security risks, especially when the code being executed is derived from untrusted sources. Malicious code can perform unintended actions, such as:

- Accessing or modifying files
- Running system commands
- Exfiltrating data
- Exploiting other parts of the system

Simply put, with exec(), an attacker can execute almost any Python code they wish, potentially leading to a complete system compromise.

#### Mitigation with 'ast.parse'
The ast.parse method from Python's Abstract Syntax Tree (AST) module provides a mechanism to analyze and modify code before execution. A properly designed inspection using ast.parse can:

- Whitelist specific operations, ensuring only safe operations are permitted.
- Blacklist or remove known-dangerous operations or patterns.
- Transform potentially unsafe constructs into safe equivalents.

By inspecting and potentially modifying the parsed AST before executing it, one can impose constraints and policies that significantly reduce the risks associated with executing arbitrary code. However, it's crucial to implement these inspections thoroughly and accurately to ensure genuine security.

### Possible Integer Overflow
#### Problem
Integer overflow occurs when an arithmetic operation attempts to create a numeric value outside the range that can be represented within a given number of bits. For instance, an 8-bit unsigned integer can represent values from 0 to 255. If this integer is incremented from 255, it wraps around to 0 rather than producing 256. This can lead to unexpected results, loss of precision and can may introduce security vulnerabilities.

#### Mitigation
Mitigating integer overflow is non-trivial because:

- Detection is Hard: Before a calculation, it's difficult to predict if the result will exceed the datatype's bounds without actually performing the calculation.
- Performance Concerns: Continuously checking for overflow can introduce overhead, which might slow down calculations, especially in loops or intensive computations.
- Different Behavior across Platforms: The way overflow is handled can vary between platforms and compilers, leading to inconsistencies.

Possible Solutions

- Wrap Method: Accept that overflow might occur and design the system to work correctly even when values wrap around.
- Error Handling: Proactively check for conditions that could result in overflow and throw an error or handle it gracefully.