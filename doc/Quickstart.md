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

# Quickstarting DAPHNE

These reduced instructions should get you started by firing up a hello world script from the latest binary release. 

## Download a Binary Release

Download and extract `daphne-<version>-bin.tgz` from the [release page](https://github.com/daphne-eu/daphne/releases).
Optionally choose the `daphne-cuda-<version>-bin.tgz` archive if you want to run DAPHNE with CUDA support (Nvidia Pascal 
hardware or newer and an installed CUDA SDK are required).

## Run DAPHNE

DAPHNE offers two ways to define integrated data analysis pipelines:

- [DaphneDSL](/doc/DaphneDSLLanguageRef.md) (DAPHNE's domain-specific language)
- [DaphneLib](/doc/DaphneLib.md) (DAPHNE's Python API)

For both ways, we provide lightweight run-scripts that set up the required environment (so your system's shared library loader finds the required `.so` files) and pass the provided parameters to the `daphne`/`python3` executable. 

### Running a DaphneDSL Script

In a bash (or compatible) shell, from the extracted DAPHNE directory, execute this command

```bash
./run-daphne.sh scripts/examples/hello-world.daph
```

Optionally you can activate CUDA ops by including `--cuda`:

```bash 
./run-daphne.sh --cuda scripts/examples/hello-world.daph
```

### Running a Python Script Using DaphneLib

In a bash (or compatible) shell, from the extracted DAPHNE directory, execute this command

```bash
./run-python.sh scripts/examples/daphnelib/shift-and-scale.py
```

## More Details

If you are interested in the details, you could have a look at

* the run-scripts: `run-daphne.sh` and `run-python.sh`
* the example DaphneDSL and DaphneLib scripts:
    * `scripts/examples/hello-world.daph`
    * `scripts/examples/daphnelib/shift-and-scale.py`
* the DAPHNE user configuration: `UserConfig.json`
* the DAPHNE help: `run-daphne.sh --help`

## What Next?

You might want to have a look at

- a more elaborate [getting started guide](/doc/GettingStarted.md)
- the [documentation](/doc)
- DaphneDSL and DaphneLib example scripts in `scripts/algorithms/` and `scripts/examples/`
