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

# Getting Started

This document summarizes everything you need to know to get started with using or extending the prototype.

### System Requirements

Please ensure that your development system meets the following requirements before trying to build the prototype.

**(*)**
You can view the version numbers as an orientation rather than a strict requirement.
Newer versions should work as well, older versions might work as well.

##### Operating system

| OS | distribution/version known to work (*) |
| --- | --- |
| GNU/Linux | Ubuntu 20.4.1 with kernel 5.8.0-43-generic |

##### Software

| tool/lib | version known to work (*) |
| ----------- | ----------- |
| clang | 10.0.0 |
| cmake | 3.16.3 |
| git | 2.25.1 |
| lld | 10.0.0 |
| ninja | 1.10.0 |
| pkg-config | 0.29.1 |
| python3 | 3.8.5 |
| java (e.g. openjdk) | 11 (1.7 should be fine) |
| uuid-dev |  |

##### Hardware

  - about 2.1 GB of free disk space (mostly due to MLIR/LLVM)

### Obtaining the Source Code

The prototype is based on MLIR, which is a part of the LLVM monorepo.
The LLVM monorepo is included in this repository as a submodule.
Thus, clone this repository as follows to also clone the submodule:

```bash
git clone --recursive https://gitlab.know-center.tugraz.at/daphne/prototype.git
```

Upstream changes to this repository might contain changes to the submodule (we might have upgraded to a newer version of MLIR/LLVM).
Thus, please pull as follows:

```bash
# in git >= 2.14
git pull --recurse-submodules

# in git < 2.14
git pull && git submodule update --init --recursive

# or use this little convenience script
./pull.sh
```

### Building the Prototype

Simply build the prototype using the build-script without any arguments:

```bash
./build.sh
```

When you do this the first time, or when there were updates to the LLVM submodule, this will also download and build the third-party material, which might increase the build time significantly.
Subsequent builds, e.g., when you changed something in this repository, will be much faster.

### Running the Tests

```bash
./test.sh
```

We use [catch2](https://github.com/catchorg/Catch2) as the unit test framework. You can use all [command line arguments](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#top) of catch2 with `test.sh`.

### Running the Prototype

Write a little DaphneDSL script or use `example.daphne`...

```
x = 1;
y = 2;
print(x + y);

m = rand(2, 3, 100.0, 200.0, 1.0, -1);
print(m);
print(m + m);
print(t(m));
```

... and execute it as follows: `build/bin/daphnec example.daphne`.

### Exploring the Source Code

As an **entry point for exploring the source code**, you might want to have a look at the code behind the `daphnec` executable, which can be found in `src/api/cli/daphnec.cpp`.

On the top-level, there are the following directories:

- `build`: everything generated during build (executables, libraries, generated source code)
- `doc`: documentation
- `src`: the actual source code, subdivided into the individual components of the prototype
- `test`: test cases
- `thirdparty`: required external software

### What Next?

You might want to have a look at
- the [documentation](https://gitlab.know-center.tugraz.at/daphne/prototype/-/tree/master/doc)
- the [contribution guidelines](https://gitlab.know-center.tugraz.at/daphne/prototype/-/blob/master/CONTRIBUTING.md)
- the [open issues](https://gitlab.know-center.tugraz.at/daphne/prototype/-/issues)
