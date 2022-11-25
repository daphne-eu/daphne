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

# A Few Early Example Algorithms in DaphneDSL

## Example data

To get some toy example data, you may execute the following commands from the repositories root directory:

```bash
mkdir data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv -o data/wine.csv
sed -i '1d' data/wine.csv
sed -i 's/;/,/g' data/wine.csv
echo '{"numRows": 4898, "numCols": 12, "valueType": "f64", "numNonZeros": 58776}' > data/wine.csv.meta
```

This downloads a dataset in CSV format, slightly transforms it to be readable with DAPHNE's current CSV reader, and creates a [metadata file](/doc/FileMetaDataFormat.md) for the CSV file.

## Some example invocations

Run these example invocations from the repository's root directory.
Note that all of them are toy examples on small datasets (either the one downloaded above or random data).

### Connected components

```bash
bin/daphne scripts/algorithms/components.daph n=100 e=500 C=\"outC.csv\"
```
*Does not work with vectorized execution (`--vec`) at the moment.*
<!-- error with --vec
loc(fused["scripts/algorithms/components.daph":40:17, "scripts/algorithms/components.daph":40:8, "scripts/algorithms/components.daph":40:22, "scripts/algorithms/components.daph":37:7]): error: operand #2 does not dominate this use
-->

### Gaussian Nonnegative Matrix Factorization (GNMF)

```bash
bin/daphne scripts/algorithms/gnmf.daph rank=2 n=100 e=500 W=\"outW.csv\" H=\"outH.csv\"
```
*Does not work with vectorized execution (`--vec`) at the moment.*
<!-- error with --vec:
terminate called after throwing an instance of 'std::runtime_error'
terminate called recursively
  what():  colIdx is out of bounds
Aborted (core dumped)
-->

### Linear Regression using the Direct Solve method

```bash
bin/daphne scripts/algorithms/lmDS.daph XY=\"data/wine.csv\" icpt=0 reg=0.0000001 verbose=true
```
<!-- successful with --vec -->

### Multinomial Logistic Regression using Trust Region method

```bash
bin/daphne scripts/algorithms/multiLogReg.daph XY=\"data/wine.csv\" B=\"output.csv\"
```
<!-- successful with --vec -->

### ~Principal Component Analysis (PCA)~

```bash
bin/daphne scripts/algorithms/pca.daph X=\"data/wine.csv\" K=2 center=true scale=false
```
*Does not work yet, because we still lack a kernel for `eigen()`.*

<!--
bin/daphne test/api/cli/algorithms/kmeans.daphne r=1000 f=10 c=5 i=3
bin/daphne test/api/cli/algorithms/lm.daphne r=1000 c=100
-->