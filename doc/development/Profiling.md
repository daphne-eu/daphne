<!--
Copyright 2023 The DAPHNE Consortium

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

# Profiling in DAPHNE
For a general overview of the profiling support in DAPHNE see the [user
profiling documentation](../Profiling.md).

Profiling is implemented via an instrumentation pass that injects calls to the
```StartProfiling``` and ```StopProfiling``` kernels at the start and end of
each block. In turn, the kernels call the PAPI-HL API start and stop functions.

### Known Issues / TODO:
* For scripts with multiple blocks (e.g. UDFs), the compiler will generate
  profiling passes for each block seperately instead of a single script-wide
  profiling pass.
* The profiling kernels should be exposed at the DSL / IR level, so that users
  can instrument / profile specific parts of their script. This will also need
  compiler cooperation, to make sure that the profiled bock is not rearranged /
  fused with other operations.
* To aid with the development and regression tracking of the runtime, the
  profiling kernels could also be extended to suport profiling specific
  kernels or parts of kernels.
