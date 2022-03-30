/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_IO_DAPHNEFILE_H
#define SRC_RUNTIME_LOCAL_IO_DAPHNEFILE_H

#include <cstdint>


struct DF_header {
	uint8_t version;
	uint8_t dt;
	uint64_t nbrows;
	uint64_t nbcols;
};

enum DF_data_t {reserved = 0, DenseMatrix_t = 1, CSRMatrix_t = 2, Frame_t = 3};

struct DF_body {
	uint64_t rx; // row index
	uint64_t cx; // column index
};

struct DF_body_block {
	uint32_t nbrows;
	uint32_t nbcols;
	uint8_t bt;
};

enum DF_body_t {empty = 0, dense = 1, sparse = 2, ultra_sparse = 3};

#endif
