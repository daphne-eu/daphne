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

Both kinds of test cases are expressed using [catch2](https://github.com/catchorg/Catch2), a widely used C++ **test framework**.
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

Ideally, the output looks as follows (the concrete number may have changed):

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

- catch2 also supports complex combinations and set operations on test names and tags.
    See the [catch2 documentation](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#specifying-which-tests-to-run) for details.

#### Additional Useful Flags

See the [catch2 documentation](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md) for a full reference of all command-line arguments.

- **Show the progress of the test execution:**

    ```bash
    ./test.sh -d yes
    ```

    By this catch2 flag, the duration of each test is displayed, which also means some progress indication.

#### DAPHNE-specific Flags

The test script has several flags that control how the DAPHNE system and the test executable are built through the build script `build.sh`.

- `--cuda`, `--fpgaopencl`, and `--mpi` switch on the respective feature of DAPHNE.
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

#### Tests Fail in the CI Workflow (even though they pass in your local development environment)

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

*Coming soon!*