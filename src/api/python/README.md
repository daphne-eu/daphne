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

Refer to the online documentation:

- [DaphneLib](https://github.com/m-birke/daphne/tree/main/scripts/examples/daphnelib)
- [Container Usage](https://daphne-eu.github.io/daphne/GettingStarted/#2-download-the-daphne-development-container-image)
- [Building DAPHNE](https://daphne-eu.github.io/daphne/development/BuildingDaphne/)

## Setup

The environment variable `DAPHNELIB_DIR_PATH` must be set to the directory with `libdaphnelib.so` and `libAllKernels.so` in it.

```sh
export DAPHNELIB_DIR_PATH='pathto/daphne/lib'
```

## Usage

More script examples on [github](https://github.com/m-birke/daphne/tree/main/scripts/examples/daphnelib)

```python
from daphne.context.daphne_context import DaphneContext
import numpy as np

dc = DaphneContext()

# Create data in numpy.
a = np.arange(8.0).reshape((2, 4))

# Transfer data to DaphneLib (lazily evaluated).
X = dc.from_numpy(a)

print("How DAPHNE sees the data from numpy:")
X.print().compute()

# Add 100 to each value in X.
X = X + 100.0

# Compute in DAPHNE, transfer result back to Python.
print("\nResult of adding 100 to each value, back in Python:")
print(X.compute())
```

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

## Publish

Use [twine](https://twine.readthedocs.io/en/stable/) for publishing to [PyPI](https://pypi.org/project/daphne-lib/). Install via `pip install twine`.

1. Set `version` in `pyproject.toml`
1. Build according to __Build__ section
1. `twine check dist/daphne-<version>-py3-none-any.whl`
    - checks the wheel file
1. `twine upload -u __token__ dist/daphne-<version>-py3-none-any.whl`
    - to publish to PyPI
    - twine prompts for your PyPI token
