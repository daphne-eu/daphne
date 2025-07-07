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

# Recording Data Properties

There is an experimental feature for recording the true data properties of intermediate results at run-time and re-inserting them in a subsequent run of DAPHNE.

By default, this feature is turned off.
Recording data properties can be turned on by invoking `daphne` with `--enable-property-recording`.
Inserting data properties recorded in a previous execution of `daphne` can be turned on with `--enable-property-insert`.
These two options are mutually exclusive.
The recorded true data properties are stored to/loaded from a simple JSON file whose path is specified by `--properties-file-path` (default: `properties.json`).
See also `daphne --help` for some help on these options.

## Possible Uses of this Feature

- Recording: Detailed insights into the data properties of intermediate results to find potential for optimizations by exploiting these data properties.
- Insertion: Find out how DAPHNE performs when it has precise knowledge of the true data properties (no unknowns, no inaccurate compile-time estimates).
- ...

## Current Limitations

- Only one data property is recorded: sparsity; more properties could be added in the future.
- Only matrix-typed intermediate results are considered; scalars and frames could be added in the future.
- Only intermediates produced in the `main` function of DaphneIR are considered; intermediates inside UDFs could be considered in the future.
- Control-flow constructs like if-then-else and loops are viewed as black boxes, i.e., the data properties of their results are recorded/inserted, but not those of intermediates created *inside* these constructs; in the future, we could consider those, too.
- ...