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

# Read and Write Data

Reading and writing (meta) data in Daphne.

When loading data with ``read()`` in a DaphneDSL script, the system expects a file with the same file name in the same
directory as the data file with an additional extension ``.meta``. This file contains a description of meta data stored
in JSON format.

There are two slightly varying ways of specifying meta data depending on whether there is a schema for the columns (e.g.,
a data frame - the corresponding C++ type is the Frame class) or not (this data can currently (as of version 0.1) be
loaded as `DenseMatrix<VT>` or `CSRMatrix<VT>` where `VT` is the value type template parameter).

If data is written from a DaphneDSL script via ``write()``, the meta data file will be written to the corresponding ``filename.meta``.

## Currently supported JSON fields

| Name        | Expected Data | Allowed values                                                                                                                                                                                                                                                                                                                               |
|-------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| numRows     | Integer       | # number of rows                                                                                                                                                                                                                                                                                                                             |
| numCols     | Integer       | # number of columns                                                                                                                                                                                                                                                                                                                          |
| valueType   | String        | ``si8, si32, si64, // signed integers (intX_t)``<br />``ui8, ui32, ui64, // unsigned integers (uintx_t)``<br />``f32, f64, // floating point (float, double)``<br /><br/>Contained within schema this may be an empty string. In this case all columns of a data frame will have the same valueType defined outside of the schema data field |
| numNonZeros | Integer       | # number of non-zeros (optional)                                                                                                                                                                                                                                                                                                             |
| schema      | JSON          | nested elements of "label" and "valueType" fields                                                                                                                                                                                                                                                                                            |
| label       | String        | column name/header (optional, may be empty string "")                                                                                                                                                                                                                                                                                        |

## Matrix Example

The example below describes a 2 by 4 dense matrix of double precision values.

### Matrix CSV

```text
-0.1,-0.2,0.1,0.2
3.14,5.41,6.22216,5
```

### Matrix Metadata

```json
{
    "numRows": 2,
    "numCols": 4,
    "valueType": "f64",
    "numNonZeros": 0
}
```

## Data Frame Example

The example below describes a 2 by 2 Frame with signed integers in the first columns named foo and double precision 
values in the second column named bar.

### Data Frame CSV

```text
1,0.5
2,1.0
```

### Data Frame Metadata

```json
{
  "numRows": 2,
  "numCols": 2,
  "schema": [
    {
      "label": "foo",
      "valueType": "si64"
    },
    {
      "label": "bar",
      "valueType": "f64"
    }
  ]
}
```

#### Data Frame Meta Data with Default ValueType

```json
{
    "numRows": 5,
    "numCols": 3,
    "valueType": "f32",
    "schema": [
        {
            "label": "a",
            "valueType": ""
        },
        {
            "label": "bc",
            "valueType": ""
        },
        {
            "label": "def",
            "valueType": ""
        }
    ]    
}
```

### Data Frame Meta Data with Empty Labels

```json
 {
     "numRows": 5,
     "numCols": 3,
     "schema": [
         {
             "label": "",
             "valueType": "si64"
         },
         {
             "label": "",
             "valueType": "f64"
         },
         {
             "label": "",
             "valueType": "ui32"
         }
     ]
 }
```
