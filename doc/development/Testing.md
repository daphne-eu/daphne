<!--
Copyright 2024 The DAPHNE Consortium

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

# Testing

Automatic testing at various levels is an important aspect of any software project.
This document describes how we do testing in DAPHNE.

**DAPHNE's testing philosophy** is to mainly focus on end-to-end tests of the entire DAPHNE system, since that is what users care about and what, thus, ultimately matters.
However, as it can be hard to trigger all possible uses of DAPHNE's internal C++ components from [DaphneDSL](/doc/DaphneDSL/LanguageRef.md) or [DaphneLib](/doc/DaphneLib/Overview.md) scripts, we also test some of those components in isolation.

Thus, there are **two kinds of test cases** in DAPHNE:

1. *Script-level test cases*, which invoke the entire DAPHNE system with some input script file.
2. *Unit test cases*, which use individual C++ functions/classes of the DAPHNE source code.

Both kinds of test cases are expressed using [catch2](https://github.com/catchorg/Catch2), a widely used C++ **testing framework**.
We use catch2 such that we don't need to reinvent the wheel for C++ testing (plus, learning about catch2 can help DAPHNE developers also in other software projects).
Reading the [catch2 documentation](https://github.com/catchorg/Catch2/blob/devel/docs/tutorial.md) is very worthwhile to better understand testing in DAPHNE.
We use the same way to define tests for both script-level and unit tests to ensure a consistent test format and a unified test summary.

The test suite should be run by developers before committing any changes.
To ensure that the code base passes the tests, the test suite is run as part of DAPHNE's CI workflows on GitHub for every commit on the main branch and every pull request to the main branch.

## Running the Test Suite

To run the test suite, execute DAPHNE's **test script** from the DAPHNE root directory as follows:

```bash
# If you built the dependencies yourself (typical in a native environment):
./test.sh

# If you want to use pre-built dependencies (typical in a DAPHNE container):
./build.sh --no-deps --target run_tests && ./test.sh -nb
```

Ideally, the output looks as follows (the concrete numbers may have changed):

```text
===============================================================================
All tests passed (416199 assertions in 1580 test cases)
```

### Background

We try to make running the test suite as simple as possible in the standard case.
For that purpose, we provide a test script `test.sh`, which (1) builds virtually all DAPHNE targets, including the test executable `run_tests` (a catch2 executable), (2) sets up some environment variables, (3) handles a few DAPHNE-specific options, and (4) invokes the test executable.
Due to this sequence of actions, invoking the test executable `run_tests` directly is discouraged.

### Customizing the Test Execution

The test execution can be controlled in several ways.
On the one hand, *all* [catch2 arguments](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md) are supported, which already yields quite some flexibility (below, we only comment on the most useful ones in DAPHNE).
On the other hand, there a few DAPHNE-specific options.
Both kinds of options can simply be added after `test.sh` and can be freely mixed.

#### Controlling which Tests to Run

The set of test cases to run can be tailored in a fine-grained way, and we completely rely on catch2's features here.
Each test case has a *name* and one or multiple *tags* (think of groups/categories of test cases).

- **Run all test cases:**

    ```bash
    ./test.sh
    ```

    If nothing special is specified, all test cases are run.

- **Run a specific test case:**

    ```bash
    ./test.sh kmeans
    ```

    As our test case names often have slightly technical-looking suffixes (due to `TEMPLATE_TEST_CASE` and `TEMPLATE_PRODUCT_TEST_CASE`, see below), it is quite handy to use wildcards:

    ```bash
    ./test.sh matmul*
    ```

- **Run all test cases with a specific tag:**

    ```bash
    ./test.sh [algorithms]
    ```

    A list of all test tags can be found in [`test/tags.h`](/test/tags.h).

- catch2 also supports **complex combinations and set operations** on test names and tags.
    See the [catch2 documentation](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#specifying-which-tests-to-run) for details.

#### Additional Useful Flags

See the [catch2 documentation](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md) for a full reference of all command-line arguments.

- **Show the progress of the test execution:**

    ```bash
    ./test.sh -d yes
    ```

    By this catch2 flag, the duration of each test is displayed, which also means some progress indication.

#### DAPHNE-specific Flags

The test script has several flags that control how DAPHNE and the test executable are built through the build script `build.sh`.

- `--cuda`, `--fpgaopencl`, and `--mpi` switch on the respective feature of DAPHNE.
- `--no-papi` switch off the respective feature of DAPHNE.
- `--debug` builds DAPHNE with debug information.
- `-nb` or `--no-build` builds DAPHNE with pre-compiled dependencies (available in the DAPHNE containers).

These flags are merely forwarded to `build.sh`.
Further information can be found there.

### Reacting to Test Failures

#### Tests Fail in Your Local Development Environment

Finding out what's going wrong and how to fix it requires the typical debugging skills, where most developers have their own ways and preferences.

Some suggestions in the context of DAPHNE:

- Look at the error messages produced by catch2, which indicate the name of the failing test case as well as the assertions that failed (including detailed information like the expected and found values). This output is often sufficient to narrow down the problem.
- Try to change the code and/or the test case and execute only the affected test case separately (see above) to save time. For script-level test cases, it can be helpful to invoke the script outside the test suite (the scripts typically reside in `test/api/cli/`, make sure to use the same arguments as in the test suite).
- Use [DAPHNE's logger](/doc/development/Logging.md) or custom print-outs to generate more debug output.
- Use a debugger.

#### Tests Fail in the CI Workflow (Even Though They Pass in Your Local Development Environment)

Unfortunately, this can happen.
**Possible reasons** include:

- **Differences in the software setups of your system and the CI system.**
    The CI machine may have a different OS (version), C++ compiler (version), installed software packages (versions), etc.
    You can try to debug the failing tests in the CI's *OS/software* environment by using the CI container image as follows:

    *On the host (from the DAPHNE root directory):*

    ```bash
    docker pull daphneeu/github-action
    docker run -it --rm -w /daphne -v "$(pwd):/daphne" daphneeu/github-action:latest bash
    ```

    *In the container:*
    
    ```bash
    # To avoid cmake complaints.
    ./build.sh --clean -y
    # Build the test cases (effectively all targets) using pre-built dependencies.
    ./build.sh --no-deps --target run_tests
    # Run the tests and print which test is running.
    ./test.sh -nb -d yes
    ```

    If the test failures cannot be reproduced that way, they are most likely not due to OS/software environment reasons.

- **Differences in the hardware setups of your system and the CI system.**
    The CI machine may have a different processor model, a different number of physical/virtual cores, a different memory/disk capacity etc.
    These differences could potentially have an impact on race conditions, the numerical stability of results, etc.
    Ideally, DAPHNE should behave well in a reasonable range of hardware setups.

## Writing Test Cases

### Overview of the Testing Code

All testing code resides in the directory `test/`.
The subdirectories of `test/` are (almost) the same as those of the source code directory `src/` and contain the test cases for the respective parts of the code base.

- `test/`
    - `api/`: all script-level test cases
        - `cli/`: command-line use of DAPHNE (DaphneDSL scripts)
        - `python`: Python API use of DAPHNE (DaphneLib Python scripts)
    - `data/`: data sets to be used in script-level test cases
    - `codegen/`, `ir/`, `parser/`, `runtime/` (all remaining subdirectories): all unit test cases
    - `CMakeLists.txt`: list of all `cpp`-files containing test cases
    - `run_tests.h`/`.cpp`: source code of the test executable `run_tests` (including the catch2 main function), *rarely needs editing*
    - `tags.h`: list of all test case tags

In the following, we give an **overview on writing test cases** in DAPHNE.
The main goal is to enable developers to understand the way we write test cases in DAPHNE, rather than providing a detailed tutorial.
***There is already a multitude of existing test cases that can serve as concrete examples to learn from.***

**To add a new test case**, first identify the right spot in the directory hierarchy.
Then, either create a new `*Test.cpp`-file (and add it to `test/CMakeLists.txt`) or extend an existing one.
Within this `*Test.cpp`-file, you can add any logic required to perform your test, including helper functions etc.
Create the individual [catch2 test cases](https://github.com/catchorg/Catch2/blob/devel/docs/test-cases-and-sections.md) with macros like `TEST_CASE`, `TEMPLATE_TEST_CASE` and `TEMPLATE_PRODUCT_TEST_CASE`.
Inside a test case, you can implement whatever you need to (a) set up the test fixture, (b) invoke the components to test, and (c) free any allocated resources.
While test cases can be totally custom, we comment on frequent cases below.

### Writing Script-level Test Cases

**Script-level test cases** test the end-to-end use of DAPHNE by invoking DaphneDSL or DaphneLib scripts like users do.
That way, we can involve the entire DAPHNE parser, compiler, and runtime stack into the test case.
To that end, we invoke the `daphne` executable (DaphneDSL scripts) or `python3` (DaphneLib scripts) as separate processes from our C++ test cases, pass them all required arguments, and capture their status code as well as their output to `stdout` and `stderr`.
We typically check if a given DaphneDSL/DaphneLib script:

1. parses, compiles, and executes successfully (or fails in the expected way) (status code)
2. yields the expected output (`stdout` and `stderr`)

We offer a **hierarchy of utilities** for common tasks in `test/api/cli/Utils.h`/`.cpp`.
These utilities work at slightly different abstraction levels.
In case there is no utility function that exactly solves your case, you can still compose the lower-level utlities.
In the following, we list just the most important ones, see the mentioned file for a detailed reference.

- **`runProgram`**`(std::stringstream & out, std::stringstream & err, const char * execPath, Args ... args)`

    Executes the specified program with the given arguments and captures `stdout`, `stderr`, and the status code.

    This is the basis for the other utility functions.

- **`runDaphne`**`(std::stringstream & out, std::stringstream & err, Args ... args)`

    Executes DAPHNE's command line interface with the given arguments and captures `stdout`, `stderr`, and the status code.

- **`runDaphneLib`**`(std::stringstream & out, std::stringstream & err, const char * scriptPath, Args ... args)`

    Executes the given Python script with the `python3` interpreter and captures `stdout`, `stderr`, and the status code.

- **`runLIT`**`(std::stringstream &out, std::stringstream &err, std::string dirPath,            Args... args)`

    Executes the "run-lit.py" Python script in a directory and captures `stdout`, `stderr`, and the status code.

    Used for tests of `mlir`-files.

- **`checkDaphneStatusCode`**`(StatusCode exp, const std::string & scriptFilePath, Args ... args)`

    Checks whether executing the given DaphneDSL script with the command line interface of DAPHNE returns the given status code.

- **`checkDaphneFails`**`(const std::string & scriptFilePath, Args ... args)`

    Checks whether executing the given DaphneDSL script with the command line interface of DAPHNE fails.

- **`compareDaphneToStr`**`(const std::string & exp, const std::string & scriptFilePath, Args ... args)`

    Compares the standard output of executing the given DaphneDSL script with the command line interface of DAPHNE to a reference text.

- **`compareDaphneToRef`**`(const std::string & refFilePath, const std::string & scriptFilePath, Args ... args)`

    Compares the standard output of executing the given DaphneDSL script with the command line interface of DAPHNE to a reference text file.

- **`compareDaphneToDaphneLib`**`(const std::string & pythonScriptFilePath, const std::string & daphneDSLScriptFilePath, Args ... args)`
    
    Compares the standard output of the given DaphneDSL script with that of the given Python/DaphneLib script.

- **`compareDaphneToSelfRef`**`(const std::string &expScriptFilePath, const std::string &actScriptFilePath, Args ... args)`

    Compares the standard output of executing a given DaphneDSL script with the command line interface of DAPHNE to a (simpler) DaphneDSL script defining the expected behavior.

**Some more hints:**

- The `.cpp`-files of the DaphneDSL script-level test cases often define macros like `MAKE_TEST_CASE` and `MAKE_FAILURE_TEST_CASE`. These wrap the test case creation such that we can easily create families of similar test cases.
- Many of the script-level test cases have no highly descriptive names as that would be cumbersome. Instead, they are typically numbered. In that context, several of the utility functions mentioned above have a variant suffixed with `Simple` (e.g., `compareDaphneToRefSimple()`), which assumes a certain naming scheme of the test cases.
- Please put a comment line at the top of each DaphneDSL/DaphneLib script used in a script-level test case that briefly explains what is being tested. Note that the purpose of the test is otherwise not always obvious. Putting a comment helps to keep the test case up-to-date when DAPHNE changes.
- When using hand-written reference text files (e.g., with `compareDaphneToRef()`), note that there usually needs to be a newline at the end of the text file, as the DAPHNE output typically also ends with a newline. Otherwise, the test case fails even though the actual and expected outputs look quite similar at first glance.

### Writing Unit Test Cases

Unit test cases invoke individual C++ components of DAPHNE in isolation to test their behavior in various situations.
We use unit tests mainly for internal data structures, kernels, file readers/writers, as well as some pieces of the DAPHNE compiler.

#### Unit Tests for Kernels

Unit tests for kernels are the most frequent kind of unit test case in DAPHNE.
These tests reside in `test/runtime/local/kernels/`.
For each DaphneIR operation `Xyz` (i.e., per `Xyz.h` in `src/runtime/local/kernels/`), there is one `XyzTest.cpp`-file.
[DAPHNE's kernels](/doc/development/ImplementBuiltinKernel.md) make extensive use of C++ template metaprogramming.
Thus, the test cases are usually defined by catch2's `TEMPLATE_TEST_CASE` or `TEMPLATE_PRODUCT_TEST_CASE`, but some also use the plain `TEST_CASE`.

The **typical structure** of a unit test case for a kernel is as follows:

1. *Define multiple pairs of inputs and expected outputs.* These should include various regular and corner cases as well as cases with invalid arguments. [catch2 SECTION](https://github.com/catchorg/Catch2/blob/devel/docs/test-cases-and-sections.md)s can be used to define such cases very concisely, with minimal code duplication.
2. *Call the kernel under test as a C++ function* with the given inputs and obtain the outputs.
3. *Compare the actual and expected outputs.* To this end, catch2 offers various [assertion macros](https://github.com/catchorg/Catch2/blob/devel/docs/assertions.md), such as `CHECK`, `REQUIRE`, `CHECK_FALSE`, `CHECK_THROWS` etc. In DAPHNE, the `==` operator can be used to compare entire DAPHNE matrices and frames, e.g., `*exp == *res`. The `checkEqApprox`-kernel can be used to take floating-point round-off errors into account when comparing DAPHNE data objects.
4. *Destroy all created data objects exactly once* using `DataObjectFactory::destroy()`.

For steps 2 and 3, we often employ a separate `checkXyz()` helper function.

**Some more hints:**

- Don't encode the types provided by `TEMPLATE_TEST_CASE` and `TEMPLATE_PRODUCT_TEST_CASE` in the test name manually; catch2 appends them automatically.
- The type of a particular instantiation of a `TEMPLATE_TEST_CASE` or `TEMPLATE_PRODUCT_TEST_CASE` is available as `TestType` within the test case. DAPHNE matrices further expose their value type as a member type `VT` (e.g., `TestType::VT`) and offer `withValueType<VT>` as a utility for getting the same data type with a different value type (e.g., `TestType::WithValueType<double>`). Furthermore, utilities from the STL header `<type_traits>` can be very helpful to manipulate C++ types.
- Small test matrices with hardcoded elements can easily be created by `genGivenVals()` (see `src/runtime/datagen/GenGivenVals.h`).
- All kernels (except for `createDaphneContext`) expect a `DaphneContext` as their last parameter. In the context of an invocation of the `daphne` executable, the `DaphneContext` is normally provided by the DAPHNE compiler/runtime. In unit test cases, it is typically fine to simply pass a `nullptr` as the context.
- Try to write the checks in a way such that catch2 produces helpful outputs in case of a failure. For instance, to check if a string `s` is empty, don't use `CHECK(s.empty())`, but rather do `CHECK(s == "")`, since the latter will include the contents of `s` in the failure indication, which is usually quite helpful. Furthermore, be aware that `REQUIRE` stops the test case execution on a failure, which also means that the following checks (which might produce helpful error indications) are not performed. Thus, consider using `CHECK` instead and use `REQUIRE` only when the following checks would not even be well-defined (e.g., you could require that something is not a `nullptr`).

## Limitations and Outlook

DAPHNE's test suite is continuously under development and contributions are always welcome.
Open topics include, but are not limited to:

- **Better test coverage**
    - *Unit tests for kernels:*
        - Systematic tests with view as inputs
        - Systematic tests with zero-row/column inputs
        - Systematic tests with invalid inputs
    - *Script-level tests*
        - Systematic tests with all combinations of arguments
        - DSL fuzzing for testing a multitude of valid DaphneDSL/DaphneLib scripts
- **Comparison to baseline systems** to check correctness for complex scripts beyond hand-written expected results
- **Performance regression tests** (see #208)
- **Specification of test cases**: more concise ways, especially for many small script-level test cases