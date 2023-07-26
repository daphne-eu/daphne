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

# Dml-to-DaphneDSL Translator

SystemDS has a variety of builtin dml scripts. This translator aims to convert a given dml script to an equivalent daphneDSL script. Related to issue: [dml2daphnedsl](/issues/529)

## Running the translator

The translator can be found in `tools/dml2daph.py`. In order to run the script the path to the `.dml` file has to be provided as an argument. Converting "shortestPath.dml" to daphneDSL would look like this:

```cpp
python3 dml2daph.py ../thirdparty/systemds/scripts/builtin/shortestPath.dml
```

The translated script can then be found in `tools/translated_files/shortestPath.daph`.

## Tests

The tests can be found in `tools/tests/`. Each test folder contains a python script as well as `.dml` and `.daphne` files needed for calling the scripts.

Running the test for shortestPath in `tools/tests/test_shortestPath` would look like this:

```cpp
python3 test_shortestPath.py
```

After running the tests the folders `data` and `output` are created. The `data` folder contains the input matrices. The `output` folder contains the output matrices for both the dml script and the translated daphneDSL script.

### Implemented Tests

* sigmoid
* shortestPath
* lm

## Todo

The translator works for a variety of dml scripts, but it is still missing certain functionalities.

### Not Yet Implemented

* visitImportStatement()
    * can be implemented similar to handleImplicitImport()
* dynamically retrieving parameter types and return types of imported functions
* extend mappings from native dml functions to equivalent functions in daphne
* extend type conversion

