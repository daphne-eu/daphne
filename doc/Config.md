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

# Configuration - Getting Information from the User

The behavior of the DAPHNE system can be influenced by the user by means of a cascading configuration mechanism.
There is a set of options that can be passed from the user to the system.
These options are collected in the `DaphneUserConfig` ([src/api/cli/DaphneUserConfig.h](/src/api/cli/DaphneUserConfig.h)).
The cascade consists of the following steps:

- The defaults of all options are hard-coded directly in the `DaphneUserConfig`.
- At program start-up, a configuration file is loaded, which overrides the defaults **(WIP, #141)**.
- After that, command-line arguments further override the configuration ([src/api/cli/daphne.cpp](/src/api/cli/daphne.cpp)).
- (In the future, DaphneDSL will also offer means to change the configuration at run-time.)

The `DaphneUserConfig` is available to all parts of the code, including:

- The DAPHNE compiler: The `DaphneUserConfig` is passed to the `DaphneIrExecutor` and from there to all passes that need it.
- The DAPHNE runtime: The `DaphneUserConfig` is part of the `DaphneContext`, which is passed to all kernels.

Hence, information provided by the user can be used to influence both, the compiler and the runtime.
*The use of environment variables to pass information into the system is discouraged.*

## How to extend the configuration?

If you need to add additional information from the user, you must take roughly the following steps:

1. Create a new member in `DaphneUserConfig` and hard-code a reasonable default.
2. Add a command-line argument to the system's CLI API in [src/api/cli/daphne.cpp](/src/api/cli/daphne.cpp). We use LLVM's [CommandLine 2.0 library](https://llvm.org/docs/CommandLine.html) for parsing CLI arguments. Make sure to update the corresponding member the `DaphneUserConfig` with the parsed argument.
3. *For compiler passes*: If necessary, pass on the `DaphneUserConfig` to the compiler pass you are working on in [src/compiler/execution/DaphneIrExecutor.cpp](/src/compiler/execution/DaphneIrExecutor.cpp). *For kernels*: All kernels automatically get the `DaphneUserConfig` via the `DaphneContext` (their last parameter), so no action is required from your side.
4. Access the new member of the `DaphneUserConfig` in your code.
