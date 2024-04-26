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

## System Requirements

Please ensure that your development system meets the following requirements before trying to build the system.

**(*)**
You can view the version numbers as an orientation rather than a strict requirement.
Newer versions should work as well, older versions might work as well.

### Operating system

| OS         | distribution/version known to work (*) | Comment                                                                    |
|------------|----------------------------------------|----------------------------------------------------------------------------|
| GNU/Linux  | Manjaro                                | Last checked in January 2023                                               ||
| GNU/Linux  | Ubuntu 20.04 - 22.10                   | All versions in that range work. 20.04 needs CMake installed from Snap.    |
| GNU/Linux  | Ubuntu 18.04                           | Used with Intel PAC D5005 FPGA, custom toolchain needed                    |
| MS Windows | 10 Build 19041, 11                     | Should work in Ubuntu WSL, using the provided Docker images is recommended |

#### Windows

Installing WSL and Docker should be straight forward using the documentation proveded by [Microsoft](https://learn.microsoft.com/en-us/windows/wsl/). On an installed WSL container
launching DAPHNE via Docker (see below) should work the same way as in a native installation.

### Software

| tool/lib                             | version known to work (*)    | comment                                                                                                                                 |
|--------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| GCC/G++                              | 9.3.0                        | Last checked version: 12.2                                                                                                              |
| clang                                | 10.0.0                       |                                                                                                                                         |
| cmake                                | 3.20                         | On Ubuntu 20.04, install by `sudo snap install cmake --classic` to fulfill the version requirement; `apt` provides only version 3.16.3. |
| git                                  | 2.25.1                       |                                                                                                                                         |
| libssl-dev                           | 1.1.1                        | Dependency introduced while optimizing grpc build (which used to build ssl unnecessarily)                                               |
| libpfm4-dev                          | 4.10                         | This dependency is needed for profiling support [DAPHNE-#479]                                                                           |
| lld                                  | 10.0.0                       |                                                                                                                                         |
| ninja                                | 1.10.0                       |                                                                                                                                         |
| pkg-config                           | 0.29.1                       |                                                                                                                                         |
| python3                              | 3.8.5                        |                                                                                                                                         |
| numpy                                | 1.19.5                       |                                                                                                                                         |
| pandas                               | 0.25.3                       |                                                                                                                                         |
| java (e.g. openjdk)                  | 11 (1.7 should be fine)      |                                                                                                                                         |
| gfortran                             | 9.3.0                        |                                                                                                                                         |
| uuid-dev                             |                              |                                                                                                                                         |
| llvm-10-tools                        | 10, 15                       | On Ubuntu 22.04 you may need to install a newer `llvm-*-tools` version, such as `llvm-15-tools`.                                        |
| wget                                 |                              | Used to fetch additional dependencies and other artefacts                                                                               |
| jq                                   |                              | json commandline processor used in docker image generation scripts                                                                      |
| ***                                  | ***                          | ***                                                                                                                                     |
| CUDA SDK                             | 11.7.1                       | Optional for CUDA ops                                                                                                                   |
| OneAPI SDK                           | 2022.x                       | Optional for OneAPI ops                                                                                                                 |
| Intel FPGA SDK or OneAPI FPGA Add-On | 2022.x                       | Optional for FPGAOPENCL ops                                                                                                             |
| tensorflow                           | 2.13.1                       | Optional for data exchange between DaphneLib and TensorFlow |
| torch                                | 2.3.0+cu121                  | Optional for data exchange between DaphneLib and PyTorch |

### Hardware

- about 7.5 GB of free disk space to build from source (mostly due to dependencies)
- Optional:
  - NVidia GPU for CUDA ops (tested on Pascal and newer architectures); 8GB for CUDA SDK
  - Intel GPU for OneAPI ops (tested on Coffeelake graphics); 23 GB for OneAPI
  - Intel FPGA for FPGAOPENCL ops (tested on PAC D5005 accelerator); 23 GB for OneAPI

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

If the build fails in between (e.g., due to missing packages), multiple build directories (e.g., daphne, antlr, llvm)
require cleanup. To only remove build output use the following two commands:

```bash
./build.sh --clean
./build.sh --cleanDeps
```

If you want to remove downloaded and extracted artifacts, use this:

```bash
./build.sh --cleanCache
```

For convenience, you can call the following to remove them all.

```bash
./build.sh --cleanAll
```

See [this page](/doc/development/BuildingDaphne.md) for more information.

### Setting up the environment

As DAPHNE uses shared libraries, these need to be found by the operating system's loader to link them at runtime.
Since most DAPHNE setups will not end up in one of the standard directories (e.g., `/usr/local/lib`), environment variables
are a convenient way to set everything up without interfering with system installations (where you might not even have
administrative privileges to do so).

```bash
# from your cloned DAPHNE repo or your otherwise extracted sources/binaries: 
export DAPHNE_ROOT=$PWD 
export LD_LIBRARY_PATH=$DAPHNE_ROOT/lib:$DAPHNE_ROOT/thirdparty/installed/lib:$LD_LIBRARY_PATH
# optionally, you can add the location of the DAPHNE executable to your PATH:
export PATH=$DAPHNE_ROOT/bin:$PATH
```

If you're running/compiling DAPHNE from a container you'll most probably **_*_**not**_*_** need to set these environment
variables (unless you have reason to customize your setup - then it is assumed that you know what you are doing).

### Running the Tests

```bash
./test.sh
```

We use [catch2](https://github.com/catchorg/Catch2) as the unit test framework. You can use all [command line arguments](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#top) of catch2 with `test.sh`.

### Running the DAPHNE system

Write a little DaphneDSL script or use [`scripts/examples/hello-world.daph`](/scripts/examples/hello-world.daph)...

```csharp
x = 1;
y = 2;
print(x + y);

m = rand(2, 3, 100.0, 200.0, 1.0, -1);
print(m);
print(m + m);
print(t(m));
```

... and execute it as follows: `bin/daphne scripts/examples/hello-world.daph` (This command works if Daphne is run
after building from source. Omit "build" in the path to the Daphne binary if executed from the binary distribution).

Optionally flags like ``--cuda`` can be added after the daphne command and before the script file to activate support
for accelerated ops (see [software requirements](#software) above and [build instructions](development/BuildingDaphne.md)).
For further flags that can be set at runtime to activate additional functionality, run ``daphne --help``.

### Building and Running with Containers [Alternative path for building and running the system and the tests]

If one wants to avoid installing dependencies and avoid conflicting with his/her existing installed libraries, one may
use containers.

- you need to install Docker or Singularity: Docker version 20.10.2 or higher | Singularity version 3.7.0-1.el7 or
higher are sufficient
- you can use the provided docker files and scripts to create and run DAPHNE.

**A full description on containers is available in the [containers](/containers) subdirectory.**

The following recreates all images provided by [daphneeu](https://hub.docker.com/u/daphneeu)

```bash
cd containers
./build-containers.sh
```

Running in an interactive container can be done with this run script, which takes care of mounting your
current directory and handling permissions:

```bash
# please customize this script first
./containers/run-docker-example.sh
```

For more about building and running with containers, refer (once again) to the directory `containers/` and its
[README.md](/containers/README.md).
For documentation about using containers in conjunction with our cluster deployment scripts, refer to [Deploy.md](/doc/Deploy.md).

### Exploring the Source Code

As an **entry point for exploring the source code**, you might want to have a look at the code behind the `daphne` executable, which can be found in `src/api/cli/daphne.cpp`.

On the top-level, there are the following directories:

- `bin`: after compilation, generated binaries will be placed here (e.g., daphne)
- `build`: temporary build output
- [`containers`:](/containers) scripts and configuration files to get/build/run with Docker or Singularity containers
- [`deploy`:](/deploy) shell scripts to ease deployment in SLURM clusters
- [`doc`:](/doc) documentation written in markdown (e.g., what you are reading at the moment)
- `lib`: after compilation, generated library files will be placed here (e.g., libAllKernels.so, libCUDAKernels.so, ...)
- [`scripts`:](/scripts) a collection of algorithms and examples written in DAPHNE's own domain specific language ([DaphneDSL](/doc/DaphneDSL/LanguageRef.md))
- [`src`:](/src) the actual source code, subdivided into the individual components of the system
- [`test`:](/test) test cases
- `thirdparty`: required external software

### What Next?

You might want to have a look at

- the [documentation](/doc/)
- the [contribution guidelines](/CONTRIBUTING.md)
- the [open issues](https://github.com/daphne-eu/daphne/issues)
