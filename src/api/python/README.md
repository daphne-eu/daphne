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

# DaphneLib

Refer to the [online docs](https://daphne-eu.github.io/daphne/) for documentation.

## Setup

The environment variable `DAPHNELIB_DIR_PATH` must be set to the directory with `libdaphnelib.so` and `libAllKernels.so` in it.

## Build

Build Python wheel package:

```sh
pip install build
./clean.sh && python3 -m build --wheel
```

## Dev Setup

With editable install

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
