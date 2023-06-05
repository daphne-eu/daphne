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

# Quickstart

These reduced instructions should get you started by firing up a hello world script from the latest binary release.

**The recipe is as follows:**

1. Download and extract `daphne-<version>-bin.tgz` from the [release page](https://github.com/daphne-eu/daphne/releases).
Optionally choose the `daphne-cuda-<version>-bin.tgz` archive if you want to run DAPHNE with CUDA support (Nvidia Pascal
hardware or newer and an installed CUDA SDK are required)
2. In a bash (or compatible) shell, from the extracted DAPHNE directory, execute this command

    ```bash
    ./run-daphne.sh scripts/examples/hello-world.daph
    ```

    Optionally you can activate CUDA ops by including --cuda:

    ```bash
    ./run-daphne.sh --cuda scripts/examples/hello-world.daph
    ```

    Earning extra points: To see one level of intermediate representation that the DAPHNE compiler generates in its wealth of optimization passes run with the explain flag

    ```bash
    ./run-daphne.sh --explain=kernels scripts/examples/hello-world.daph
    ```

## Explanation

The ``run-daphne.sh`` script sets up the required environment (so your system's shared library loader finds the required
.so files) and passes the provided parameters to the daphne executable.

Interesting things to look at:

- file ``run-daphne.sh``
- file ``UserConfig.json``
- file ``scripts/examples/hello-world.daph``
- output of ``run-daphne.sh --help``

### What Next?

You might want to have a look at

- a more elaborate [getting started guide](/doc/GettingStarted.md)
- the [documentation](/doc)
- the [contribution guidelines](/CONTRIBUTING.md)
- the [open issues](https://github.com/daphne-eu/daphne/issues)
