# DaphneLib

Refer to [online docs](https://daphne-eu.github.io/daphne/) for documentation.

## Setup

Env var `DAPHNELIB_DIR_PATH` must be set to dir with `libdaphnelib.so` and `libAllKernels.so` in it.

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
