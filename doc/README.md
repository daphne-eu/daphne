<!--
Copyright 2025 The DAPHNE Consortium

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

# Overview

- [Getting Started](/doc/GettingStarted.md)

## User Documentation

- For the `daphne` command-line interface API: see `bin/daphne --help`
- [Running DAPHNE in a Local Environment](/doc/RunningDaphneLocally.md)
- [Running DAPHNE on the Distributed Runtime](/doc/DistributedRuntime.md)
- [DAPHNE Packaging, Distributed Deployment, and Management](/doc/Deploy.md)
- [DaphneLib: DAPHNE's Python API](/doc/DaphneLib/Overview.md)
- [DaphneLib API Reference](/doc/DaphneLib/APIRef.md)
- [Porting Numpy to DaphneLib](/doc/DaphneLib/Numpy2DaphneLib.md)
- [DaphneDSL Language Reference](/doc/DaphneDSL/LanguageRef.md)
- [DaphneDSL Built-in Functions](/doc/DaphneDSL/Builtins.md)
- [Using SQL in DaphneDSL](/doc/tutorial/sqlTutorial.md)
- [A Few Early Example Algorithms in DaphneDSL](/scripts/algorithms/)
- [FileMetaData Format (reading and writing data)](/doc/FileMetaDataFormat.md)
- [Profiling DAPHNE using PAPI](/doc/Profiling.md)
- [Custom Extensions to DAPHNE](/doc/Extensions.md)
- [HDFS Usage](/doc/HDFSUsage.md)

## Developer Documentation

* [Creating release artifacts](ReleaseScripts.md)
* [Maintaining GPG signing keys](GPG-signing-keys.md)
* [Data Properties: Representation, Inference, and Exploitation](/doc/development/PropertyInference.md)
* [Recording Data Properties](/doc/development/PropertyRecording.md)
* [Code Generation with MLIR](/doc/Codegen.md)

### How-tos and Guidelines

- [Handling a Pull Request](/doc/development/HandlingPRs.md)
- [Implementing a Built-in Kernel](/doc/development/ImplementBuiltinKernel.md)
- [Binary Data Format](/doc/BinaryFormat.md)
- [DAPHNE Configuration: Getting Information from the User](/doc/Config.md)
- [Extending DAPHNE with more scheduling knobs](/doc/development/ExtendingSchedulingKnobs.md)
- [Extending the DAPHNE Distributed Runtime](/doc/development/ExtendingDistributedRuntime.md)
- [Building DAPHNE with the build.sh script](/doc/development/BuildingDaphne.md)
- [Building in Unsupported Environments](/doc/BuildEnvironment.md)
- [Logging in DAPHNE](/doc/development/Logging.md)
- [Profiling in DAPHNE](/doc/development/Profiling.md)
- [Testing](/doc/development/Testing.md)
- [Installing Python Libraries in the `daphne-dev` Container](/doc/development/InstallPythonLibsInContainer.md)

### Source Code Documentation

The source code is partly documented in doxygen style.
Until the generation of a proper documentation has been set up, you can simply have a look at the documentation in the individual source files.
