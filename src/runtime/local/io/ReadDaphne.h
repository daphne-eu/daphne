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

#pragma once

#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <runtime/local/io/utils.h>

#include <util/preprocessor_defs.h>

#include <type_traits>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdlib>


// ****************************************************************************
// Struct for Daphne binary file format 
// ****************************************************************************

struct DF_header {
	uint8_t version;
	uint8_t dt;
	uint64_t nbrows;
	uint64_t nbcols;
};

enum DF_data_t {reserved = 0; DenseMatrix = 1; CSMatrix = 2; Frame = 3};
// Value types defined in /runtime/local/datastructures/ValueTypeCode.h
struct DF_body {
	uint64_t rx; // row index
	uint64_t cx; // column index
};

struct DF_body_block {
	uint32_t nbrows;
	uint32_t nbcols;
	uint8_t bt;
};

enum DF_body_t {empty = 0; dense = 1; sparse = 2; ultra-sparse = 3};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadDaphne {
  static void apply(DTRes *&res, const char *filename) = delete;

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readDaphne(DTRes *&res, const char *filename) {
  ReadDaphne<DTRes>::apply(res, filename);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template <typename VT> struct ReadDaphne<DenseMatrix<VT>> {
  static void apply(DenseMatrix<VT> *&res, const char *filename) {

    ifstream f;
    f.open(filename);
    // TODO: check f.good()

    // read header
    DF_header h;
    f.read(h, sizeof(DF_header));

    if (h.dt == DF_data_t::DenseMatrix || h.dt == DF_data_t::CSRMatrix) {
	    uint8_t vt;
	    f.read(vt, size of vt);

	    DF_body b;
	    f.read(b, sizeof(DF_body));
	    // b is ignored for now - assumed to be 0,0
	    //TODO: consider multiple blocks

	    DF_body_block bb;
	    f.read(bf,sizeof(DF_body_header));
	    if (bb.bb == DF_body_t::empty) {
		res = DataObjectFactory::create<DenseMatrix<VT>>(0, 0, false);
		return;

	    // Dense Matrix
	    } else if (bb.bt == DF_body_t::dense) {
		 uint8_t vt;
		 f.read(vt, size of vt);

		 std::shared_ptr<VT[]> memblock = new VT[bb.numrows*bb.numcols];
		 f.read(vt, size of memblock);

		 res = DataObjectFactory::create<DenseMatrix<VT>>(bb.numrows, bb.numcols, memblock, false);

		 goto exit;

	    // CSR Matrix
	    } else if (bb.bt == DF_body_t::sparse) {
		    uint8_t vt;
		    f.read(vt, size of vt);

		    uint64_t nzb;
		    f.read(nzb, size of nzb);

		    res = DataObjectFactory::create<CSRMatrix<VT>>(
				bb.numrows, bb.numcols, nzb, false);

		    for (int i = 0; i < bb.nbrows; i++) {
			    uint64_t nzr;
			    f.read(nzr, size of nzr);

			    for (n = 0; n < nzr; n++) {
				  uint32_t j;
				  f.read(j, size of j);

				  VT val;
				  f.read(val, size of val);

				  res->set(i, j, val);
			    }
		    }

		goto exit;

            // COO Matrix
	    } else if (bb.bt == DF_body_t::ultra_sparse) {
		    uint8_t vt;
		    f.read(vt, size of vt);

		    uint64_t nzb;
		    f.read(nzb, size of nzb);

		    res = DataObjectFactory::create<CSRMatrix<VT>>(
				bb.numrows, bb.numcols, nzb, false);

		   // Single column case
		   if (bb.numcols == 1) {
			for (int n = 0; n < nzb; n++) {
				uint32_t i;
				f.read(i, size of i);

				VT val;
				f.read(val, size of val);

				res->set(i, 1, val);	
			}
			goto exit;
		   } else {
			   // TODO: check numcols is greater than 1
			for (int n = 0; n < nzb; n++) {
				uint32_t i;
				f.read(i, size of i);

				uint32_t j;
				f.read(j, size of j);

				VT val;
				f.read(val, size of val);

				res->set(i, j, val);	
			}
			goto exit;
		   }

	    }
exit:
	    f.close();
	    return;
    }
};

