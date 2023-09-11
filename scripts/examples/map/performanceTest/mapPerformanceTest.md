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

## Table of Contents
This file explains the testing of the external map function, that can invoke map kernels of different programming languages.

- Background
- Testing
    - Utility Test
    - Performance Test
- Results
- Known Limitations
    - Safety Concerns of using exec in the Python map kernel
    - Possible Integer Overflow

## Background
DaphneDSL (DAPHNE's domain-specific language for linear and relational algebra) supports the second-order function map(). The map function expects two inputs, a matrix and a user-defined function (UDF) written in DaphneDSL. The given UDF must have a single scalar argument and return a single scalar value. map() applies the given UDF to each value in the given matrix.

As DaphneDSL is a domain-specific, rather than a general-purpose language, implementing more complex UDFs may not always be convenient or even possible. Therefore, the task was to explore how to support UDFs written in other languages. As a starting point Python was choosen as programming language.

## Testing
The tests should show the effectiveness and efficiency of the extended `map()`-kernel:

1.The utility test examines the effectiveness of the Pyton map kernel. It demonstates how this kernel can execute functions that are not feasible with pure DaphneDSL.

2.The performance test should show the efficiency of the extended `map()`-kernel in comparison to DaphneDSLs map function, elementwise operations and a Numpy approach in Python:

Metrics:
- `Memory Consumption`: Understand the footprint of the map operation.
- `Execution Time`: Gauge the responsiveness and speed.
- `CPU Load`: Measure the strain on processing resources.

Arithmetic operations of different complexity as UDFs are invoked on random generated matrices with no sparsity, a seed of 42 and different sizes.

The performance test consists of 2 python scripts: The [generateTestData.py](/scripts/examples/map/performanceTest/generateTestData.py) scripts generates the data for the different metrics and saves them in .csv-file with the headers `Run,Operation,Implementation,Datatype,MatrixSize,MetricType,Value`. The [visualizeTestData.py](/scripts/examples/map/performanceTest/visualizeTestData.py) script visualizes the data in different boxplots. You have the options of visualizing it in different .png images or generate an interactive .html file, where you can more easily display the boxplots in a browser.

Note that testcases for the general functionality are incorporated in the C++ kernel tests in the test file [AlternativeMapTest.cpp](/test/runtime/local/kernels/AlternativeMapTest.cpp).

## Known Limitations in the General Approach

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

Possible Solutions

- Wrap Method: Accept that overflow might occur and design the system to work correctly even when values wrap around.
- Error Handling: Proactively check for conditions that could result in overflow and throw an error or handle it gracefully.