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

This document summarizes everything you need to know to get started with using or extending the DAPHNE system.

### System Requirements

Please ensure that your development system meets the following requirements before trying to build the system.

**(*)**
You can view the version numbers as an orientation rather than a strict requirement.
Newer versions should work as well, older versions might work as well.

##### Operating system

| OS | distribution/version known to work (*) |
| --- | --- |
| GNU/Linux | Ubuntu 20.04.1 with kernel 5.8.0-43-generic |

##### Software

| tool/lib            | version known to work (*) | comment                                                                                                                                      |
|---------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| clang               | 10.0.0                  |                                                                                                                                              |
| cmake               | 3.17                    | On Ubuntu 20.04, install by `sudo snap install cmake --classic` to fulfill the version requirement; `apt` provides only version 3.16.3.      |
| git                 | 2.25.1                  |                                                                                                                                              |
| libssl-dev          | 1.1.1                   | Dependency introduced while optimizing grpc build (which used to build ssl unnecessarily)
| lld                 | 10.0.0                  |                                                                                                                                              |
| ninja               | 1.10.0                  |                                                                                                                                              |
| pkg-config          | 0.29.1                  |                                                                                                                                              |
| python3             | 3.8.5                   |                                                                                                                                              |
| numpy               | 1.19.5                  |                                                                                                                                              |
| java (e.g. openjdk) | 11 (1.7 should be fine) |                                                                                                                                              |
| gfortran            | 9.3.0                   |                                                                                                                                              |
| uuid-dev            |                         |                                                                                                                                              |
| libboost-dev        | 1.71.0.0 | Only required when building with support for Arrow (`--arrow`) |

##### Hardware

  - about 2.1 GB of free disk space (mostly due to MLIR/LLVM)

### Obtaining the Source Code

The DAPHNE system is based on MLIR, which is a part of the LLVM monorepo.
The LLVM monorepo is included in this repository as a submodule.
Thus, clone this repository as follows to also clone the submodule:

```bash
git clone --recursive https://github.com/daphne-eu/daphne.git
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

### Building the DAPHNE system

Simply build the system using the build-script without any arguments:

```bash
./build.sh
```

When you do this the first time, or when there were updates to the LLVM submodule, this will also download and build the third-party material, which might increase the build time significantly.
Subsequent builds, e.g., when you changed something in this repository, will be much faster.

If the build fails in between (e.g., due to missing packages), multiple build directories (e.g., daphne, antlr, llvm) require cleanup. For convenience, you can call the following to remove them all.

```bash
./build.sh --clean
```

See [this page](/doc/development/BuildingDaphne) for more information.

### Running the Tests

```bash
./test.sh
```

We use [catch2](https://github.com/catchorg/Catch2) as the unit test framework. You can use all [command line arguments](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#top) of catch2 with `test.sh`.

### Running the DAPHNE system

Write a little DaphneDSL script or use [`scripts/examples/hello-world.daph`](../scripts/examples/hello-world.daph)...

```
x = 1;
y = 2;
print(x + y);

m = rand(2, 3, 100.0, 200.0, 1.0, -1);
print(m);
print(m + m);
print(t(m));
```

... and execute it as follows: `build/bin/daphne scripts/examples/hello-world.daph`.

### Building and running with containers [Alternative path for building and running the system and the tests]
If one wants to avoid installing dependencies and avoid conflicting with his/her existing installed libraries, one may use containers.
- you need to install Docker or Singularity: Docker version 20.10.2 or higher | Singularity version 3.7.0-1.el7 or higher are sufficient
- you can use the provided docker file to create an image that contains all dependencies as follows:
```bash
cd daphne/deploy
docker build -t <ImageTag> .
#the image can be built from the dockerhub docker://ahmedeleliemy/test-workflow:latest as well
docker run -v absolute_path_to_daphne/:absolute_path_to_daphne_in_the_container -it <ImageTag> bash
[root@<some_container_ID>]cd absolute_path_to_daphne_in_the_container
[root@<some_container_ID>]./build.sh #or ./test.sh  
```
 - you can also use Singularity containers instead of docker as follows:
  ```bash
singularity build <ImageName.sif> docker://ahmedeleliemy/test-workflow
#one can also use [Singularity python](https://singularityhub.github.io/singularity-cli/)
#to convert the provided Dockerfile into Singularity recipe 
singularity shell <ImageName.sif>
Singularity> cd daphne
Singularity> ./build.sh #or ./test.sh  
```
- Because the container instance works on the same folder, if one already built the system outside the container, it is recommended to clean all build files to avoid conflicts.
- One may also do the commits from within the containers as normal.

### Exploring the Source Code

As an **entry point for exploring the source code**, you might want to have a look at the code behind the `daphne` executable, which can be found in `src/api/cli/daphne.cpp`.

On the top-level, there are the following directories:

- `build`: everything generated during build (executables, libraries, generated source code)
- `doc`: documentation
- `src`: the actual source code, subdivided into the individual components of the system
- `test`: test cases
- `thirdparty`: required external software

### What Next?

You might want to have a look at
- the [documentation](/doc)
- the [contribution guidelines](/CONTRIBUTING.md)
- the [open issues](https://github.com/daphne-eu/daphne/issues)
