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

# Running DAPHNE Locally

Running DAPHNE in a local environment.

This document explains how to run DAPHNE on a local machine.
For more details on running DAPHNE in a distributed setup, please see the documentation on the [distributed runtime](/doc/DistributedRuntime.md) and [distributed deployment](/doc/Deploy.md).

Before DAPHNE can be executed, the system must be built using `./build.sh` (for more details see [Getting Started](/doc/GettingStarted.md)).
The main executable of the DAPHNE system is `bin/daphne`.
The general scheme of an invocation of DAPHNE looks as follows:

```bash
bin/daphne [options] script [arguments]
```

Where `script` is a [DaphneDSL](/doc/DaphneDSL/LanguageRef.md) file.

*Example:*

```bash
bin/daphne scripts/examples/hello-world.daph
```

Note that the present working directory should be the root directory `daphne/` when invoking the system (this requirement will be relaxed in the future).

## Passing Script Arguments

Arguments to the DaphneDSL script can be provided as space-separated pairs of the form `key=value`.
These can the accessed as `$key` in the DaphneDSL script.

*Example:*

```bash
bin/daphne test/api/cli/algorithms/kmeans.daphne r=1000 f=20 c=5 i=10
```

*This example executes a simplified variant of the k-means clustering algorithm on random data with 1000 rows and 20 features using 5 centroids and a fixed number of 10 iterations.*

`value` must be a valid DaphneDSL literal, e.g., `key=123` (signed 64-bit integer), `key=-12.3` (double-precision floating-point), or `key="hello"` (string).
Note that the quotation marks `"` are part of the string literal, so they must be escaped on a terminal, e.g., by `key=\"hello\"`. If there are whitespaces in the string, it is necessary to surround the literal with additional quotation marks, like `key="\"hello world\""`

## Command-Line Arguments

The behavior of `daphne` can be influenced by numerous command-line arguments (the `options` mentioned above).
To see the full list of available options, invoke `bin/daphne --help`.

In the following, a few noteworthy general options are mentioned.
Note that some of the more specific options are described in the documentation pages on the respective topics, e.g., [distributed execution](/doc/DistributedRuntime.md), [scheduling](/doc/SchedulingOptions.md), [configuration](/doc/Config.md), [FPGA configuration](/doc/FPGAconfiguration.md), etc.

- **`--explain`**

    Prints the MLIR-based intermediate representation (IR), the so-called *DaphneIR*, after the specified compiler passes.
    For instance, to see the IR after parsing (and some initial simplifications) and after property inference, invoke

    ```bash
    bin/daphne --explain parsing_simplified,property_inference test/api/cli/algorithms/kmeans.daphne r=1000 f=20 c=5 i=10
    ```

- **`--vec`**

    Turns on DAPHNE's vectorized execution engine, which fuses qualifying operations into vectorized pipelines. *Experimental feature.*
  
- **`--select-matrix-repr`**

    Turns on the automatic selection of a suitable matrix representation (currently dense or sparse (CSR)). *Experimental feature.*

## Return Codes

If `daphne` terminates normally, one of the following status codes is returned:

| code | meaning | example |
| ----- | ----- | ----- |
| 0 | success | everything went well |
| 1 | parser error | a syntax error in a DaphneDSL script |
| 2 | compiler/pass error | an operation was provided with inputs of incompatible shapes |
| 3 | runtime/execution error | a kernel was invoked with invalid arguments |

## Typical Errors and Troubleshooting

### `Parser error: ...`/`Pass error: ...`/`Execution error: ...`

One of the three types of errors mentioned above occured.
In many (but not yet all) cases, there will be an error message indicating what went wrong.

*Examples:*

- **Wrong way of passing string literals as DaphneDSL script arguments.**

    ```text
    line 1:0 mismatched input 'foo' expecting {'true', 'false', INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL}
    Parser error: unexpected literal
    ```

    Maybe you tried to pass a string as an argument to a DaphneDSL script and forgot the quotation marks or they got lost.
    Pass strings as `bin/daphne script.daphne foo=\"abc\"` (not `foo=abc` or `foo="abc"`) on a terminal.

- **Missing metadata file.**
  
    ```text
    Parser error: Could not open file 'data/foo.csv.meta' for reading meta data.
    ```

    Maybe you try to read a dataset called `data/foo.csv`, but the required [metadata file](/doc/FileMetaDataFormat.md) `data/foo.csv.meta` does not exist.
  
- **Using the old file metadata format.**

    ```text
    Parser error: [json.exception.parse_error.101] parse error at line 1, column 7: syntax error while parsing value - unexpected ','; expected end of input
    ```

    Maybe you try to read a dataset with `readMatrix()` or `readFrame()` in DaphneDSL, but the file metadata file does not have the right structure. Note that we changed the initial one-line text-based format to a more human-readable [JSON-based format](/doc/FileMetaDataFormat.md).

### `JIT session error: Symbols not found: ...`

This error occurs when the execution of a DaphneDSL script requires invoking a kernel with an input/output type combination that was not pre-compiled.
The first line indicates which kernel is missing for which type combination.

*Ultimately, DAPHNE will circumvent this situation automatically by knowing which kernels were pre-compiled and utilizing only those (while employing casts to adapt the types of the arguments and results, where necessary).*

At the moment, users can try to work around this by introducing casts in the DaphneDSL script.
Developers can fix this problem by adding the respective instantiation in `src/runtime/local/kernels/kernels.json`.

*Example:*

```text
JIT session error: Symbols not found: [ _ewAdd__int32_t__int32_t__int32_t ]
JIT-Engine invocation failed: Failed to materialize symbols: { (main, { _mlir_ciface_main, _mlir_main, _mlir__mlir_ciface_main, main }) }Program aborted due to an unhandled Error:
Failed to materialize symbols: { (main, { _mlir_ciface_main, _mlir_main, _mlir__mlir_ciface_main, main }) }
Aborted (core dumped)
```

### `Failed to create MemoryBuffer for: ...`

This error occurs when `daphne` is not invoked from the repository's root directory `daphne/` as `bin/daphne`.
It will be fixed in the future (see issue #445).
In the meantime, please always invoke `daphne` from the repository's root directory `daphne/`.

*Example:*

```text
Failed to create MemoryBuffer for: lib/libAllKernels.so
Error: No such file or directory
```

*Typically followed by an error or the type `JIT session error: Symbols not found: ...`, which is described above.*
