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

# Installing Python Libraries in the `daphne-dev` Container

The `daphne-dev` container (see [GettingStarted](/doc/GettingStarted.md)) already contains all *required* dependencies for running DAPHNE. However, there can be reasons to install additional Python libraries inside the container, e.g.:

1. **To use/test DaphneLib's data exchange with TensorFlow and PyTorch.** 
    DaphneLib, DAPHNE's Python API, supports the efficient data exchange with widely-used Python libraries like numpy, pandas, TensorFlow, and PyTorch.
    Numpy and pandas are required for DaphneLib.
    Thus, they are already installed in the `daphne-dev` container.
    In contrast to that, TensorFlow and PyTorch are optional for DaphneLib; if these libraries are not installed on the system, DaphneLib cannot exchange data with them, but all remaining features still work.
    Likewise, the test cases related to the data exchange with TensorFlow and PyTorch will only run if these libraries are installed.
    As TensorFlow and PyTorch would increase the `daphne-dev` container size by several gigabytes, they are *not* included in the container.
1. **To add support for additional Python libraries in DaphneLib.**
    For instance, while implementing efficient data exchange with these additional libraries.
1. **To build integrated data analysis pipelines involving additional Python libraries.**
    For instance, for experiments.

## Installing Additional Python Libraries

Additional Python libraries are best installed in a *Python virtual environment* inside the `daphne-dev` container.
To that end, execute the following commands inside the container:

**Create a Python virtual environment and activate it:**

```bash
sudo apt update
sudo apt install python3.12-venv
python3 -m venv daphne-venv
source daphne-venv/bin/activate
```

Here, we call the virtual environment `daphne-venv`.
Feel free to choose a different name.

**Install the desired Python libraries using `pip`:**

For instance, if you want to use/test DaphneLib's efficient data transfer with widely-used Python libraries like numpy, pandas, TensorFlow, and PyTorch, install the following libraries.
Feel free to install any library you like.

```bash
pip install numpy pandas tensorflow torch
```

The libraries you install that way will be stored in the `daphne-venv` directory on the host and, thus, keep existing after you shut down the container.

## Don't Forget

Every time you enter the `daphne-dev` container, make sure to activate the Python virtual environment again:

```bash
source daphne-venv/bin/activate
```