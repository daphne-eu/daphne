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

# Binary Data Format

DAPHNE defines its own binary representation for the serialization of in-memory data objects (matrices/frames).
This representation is intended to be used by default whenever we need to transfer or persistently store these in-memory objects, e.g., for

- the data transfer in the distributed runtime
- a custom binary file format
- the eviction of in-memory data to secondary storage

*Disclaimer:* The current specification is a first draft and will likely be refined as we proceed.
At the moment, we focus on the case of a single block per data object.

**Endianess:** For now, we assume *little endian*.

**Images:** In the images below, all addresses and sizes are specified in bytes (`[B]`).

## Binary Representation of a Whole Data Object

The binary representation of a data object (matrix/frame) starts with a header containing general and data type-specific information.
The data object is partitioned into rectangular blocks (in the extreme case, this can mean a single block).
All blocks are represented individually (see binary representation of a single block below) and stored along with their position in the data object.

```text
+--------+------+
| header | body |
+--------+------+
```

### Header

The header consists of the following information:

- DAPHNE binary format version number (`1` for now) (uint8)
- data type `dt` (uint8)
- number of rows `#r` (uint64)
- number of columns `#c` (uint64)

We currently support the following **data types**:

| code | data type |
| ----- | ----- |
| `0` | *(reserved)* |
| `1` | `DenseMatrix` |
| `2` | `CSRMatrix` |
| `3` | `Frame` |

We currently support the following **value types**:

| code | C++ value type |
| ----- | ----- |
| 0 | *(reserved)* |
| 1 | `uint8_t` |
| 2 | `uint16_t` |
| 3 | `uint32_t` |
| 4 | `uint64_t` |
| 5 | `int8_t` |
| 6 | `int16_t` |
| 7 | `int32_t` |
| 8 | `int64_t` |
| 9 | `float` |
| 10 | `double` |

Depending on the data type, there are more information in the header:

*For `DenseMatrix` and `CSRMatrix`*:

- value type `vt` (uint8)

```text
addr[B]  0 0 1  1 2  9 10 17 18 18
        +---+----+----+-----+-----+
        | 1 | dt | #r |  #c |  vt |
        +---+----+----+-----+-----+
size[B]   1    1    8    8     1
```

*For `Frame`*:

- value type `vt` (uint8), for each column
- length of the label `len` (uint16) and label `lbl` (character string), for each column

```text
addr[B]  0 0 1  1 2  9 10 17 18               18+#c-1 18+#c                                         *
        +---+----+----+-----+-------+     +----------+--------+--------+     +-----------+-----------+
        | 1 | dt | #r |  #c | vt[0] | ... | vt[#c-1] | len[0] | lbl[0] | ... | len[#c-1] | lbl[#c-1] |
        +---+----+----+-----+-------+     +----------+--------+--------+     +-----------+-----------+
size[B]   1    1    8    8      1               1         2     len[0]             2       len[#c-1]
```

### Body

The body consists of a sequence of:

- a pair of
  - row index `rx` (uint64)
  - column index `cx` (uint64)
- a binary block representation

For the special case of a single block, this looks as follows:

```text
addr[B]  0 7 8 15 16       *
        +---+----+----------+
        | 0 |  0 | block[0] |
        +---+----+----------+
          8    8       *
size[B] 
```

## Binary Representation of a Single Block

A single data block is a rectangular partition of a data object.
In the extreme case, a single block can span the entire data object in both dimensions (one block per data object).

General block header

- number of rows `#r` (uint32)
- number of columns `#c` (uint32)
- block type `bt` (uint8)
- block type-specific information (see below)

```text
addr[B]  0  3 4  7 8  8 9                        *
        +----+----+----+--------------------------+
        | #r | #c | bt | block type-specific info |
        +----+----+----+--------------------------+
size[B]    4    4    1               *
```

## Block types

We define different block types to allow for a space-efficient representation depending on the data.
When serializing a data object, the block types are not required to match the in-memory representation (e.g., the blocks of a `DenseMatrix` could use the *sparse* binary representation).
Moreover, different blocks may be represented as different block types (e.g., some blocks might use the *dense* binary representation and others the *sparse* one).
We currently support the following block types:

| code | block type |
| ----- | ----- |
| `0` | *empty* |
| `1` | *dense* |
| `2` | *sparse* (CSR) |
| `3` | *ultra-sparse* (COO) |

Most block types store their value type as part of the block type-specific information.
Note that the value type used for the binary representation is not required to match the value type of the in-memory object (e.g., `DenseMatrix<uint64_t>` may be represented as a *dense* block with value type `uint8_t`, if the value range permits).
Furthermore, each block may be represented using its individual value type.

### Empty block

This block type is used to represent blocks that contain only zeros of the respective value type very space-efficiently.

Block type-specific information: *none*

```text
addr[B]  0  3 4  7 8 8
        +----+----+---+
        | #r | #c | 0 |
        +----+----+---+
size[B]    4    4   1
```
  
### Dense block

Block type-specific information:

- value type `vt` (uint8)
- values `v` in row-major (value type `vt`)

Below, `S` denotes the size (in bytes) of a single value of type `vt`.
  
```text
addr[B]  0  3 4  7 8 8 9  9 10                             10+#r*#c*S
        +----+----+---+----+---------+---------+     +---------------+
        | #r | #c | 1 | vt | v[0, 0] | v[0, 1] | ... | v[#r-1, #c-1] |
        +----+----+---+----+---------+---------+     +---------------+
size[B]    4    4   1    1      S         S                  S
```

### Sparse block (compressed sparse row, CSR)

Block type-specific information:

- value type `vt`  (uint8)
- number of non-zeros in the block `#nzb` (uint64)
- for each row
  - number of non-zeros in the row `#nzr` (uint32)
  - for each non-zero in the row
    - column index `cx` (uint32)
    - value `v` (value type `vt`)

Note that both a row and the entire block might contain no non-zeros.

Below, `S` denotes the size (in bytes) of a single value of type `vt`.

```text
                                                                 18 + 4*#r +       
addr[B]  0  3 4  7 8 8 9  9 10  17 18                             #nzb*(4+S)
        +----+----+---+----+------+--------+     +--------+     +-----------+
        | #r | #c | 2 | vt | #nzb | row[0] | ... | row[i] | ... | row[#r-1] |
        +----+----+---+----+------+--------+     +--------+     +-----------+
size[B]    4    4   1    1     8              4+#nzr[i]*(4+S)

                ______________________________________|_______________________
               /                                                              \

               +---------+----------+     +----------+     +------------------+
               | #nzr[i] | nz[i, 0] | ... | nz[i, j] | ... | nz[i, #nzr[i]-1] |
               +---------+----------+     +----------+     +------------------+
                    4         4+S              4+S                  4+S 
                     
                                  ______________|____________
                                 /                           \

                                 +----------+----------------+
                                 | cx[i, j] | v[i, cx[i, j]] |
                                 +----------+----------------+
                                       4             S
```

### Ultra-sparse block (coordinate, COO)

Ultra-sparse blocks contain almost no non-zeros, so we want to keep the overhead of the meta data low.
Thus, we distinguish blocks with a single column (where we don't need to store the column index) and blocks with more than one column.

### Blocks with a single column

Block type-specific information:

- value type `vt` (uint8)
- number of non-zeros in the block `#nzb` (uint32)
- for each non-zero
  - row index `rx` (uint32)
  - value `v` (value type `vt`)

Below, `S` denotes the size (in bytes) of a single value of type `vt`.
  
```text
addr[B]  0  3 4  7 8 8 9  9 10  13 14                         14+#nzb*(4+S)
        +----+----+---+----+------+-------+     +-------+     +------------+
        | #r | #c | 3 | vt | #nzb | nz[0] | ... | nz[i] | ... | nz[#nzb-1] |
        +----+----+---+----+------+-------+     +-------+     +------------+
size[B]    4    4   1    1     4     4+S           4+S              4+S

                                          __________|__________
                                         /                     \

                                         +-------+-------------+
                                         | rx[i] | v[rx[i], 0] |
                                         +-------+-------------+
                                             4          S
```

### Blocks with more than one column

Block type-specific information:

- value type `vt` (uint8)
- number of non-zeros in the block `#nzb` (uint32)
- for each non-zero
  - row index `rx` (uint32)
  - column index `cx` (uint32)
  - value `v` (value type `vt`)

Below, `S` denotes the size (in bytes) of a single value of type `vt`.
  
```text
addr[B]  0  3 4  7 8 8 9  9 10  13 14                         14+#nzb*(8+S)
        +----+----+---+----+------+-------+     +-------+     +------------+
        | #r | #c | 3 | vt | #nzb | nz[0] | ... | nz[i] | ... | nz[#nzb-1] |
        +----+----+---+----+------+-------+     +-------+     +------------+
size[B]    4    4   1    1     4     8+S           8+S              8+S

                                    ________________|________________
                                   /                                 \

                                   +-------+-------+-----------------+
                                   | rx[i] | cx[i] | v[rx[i], cx[i]] |
                                   +-------+-------+-----------------+
                                       4       4            S
```
