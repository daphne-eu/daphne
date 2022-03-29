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

#include <runtime/local/io/DaphneFile.h>
#include <runtime/local/io/utils.h>

#include <util/preprocessor_defs.h>

#include <type_traits>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadDaphne {
  static void apply(DTRes *&res, const char *filename) = delete;
};

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

    std::ifstream f;
    f.open(filename, std::ios::in|std::ios::binary);
    // TODO: check f.good()

    // read header
    DF_header h;
    f.read((char *)&h, sizeof(h));

    if (h.dt == DF_data_t::DenseMatrix_t) {
	    ValueTypeCode vt;
	    f.read((char *)&vt, sizeof(vt));

	    DF_body b;
	    f.read((char *)&b, sizeof(b));
	    // b is ignored for now - assumed to be 0,0
	    //TODO: consider multiple blocks

	    DF_body_block bb;
	    f.read((char *)&bb,sizeof(bb));
	    // empty Matrix
	    if (bb.bt == DF_body_t::empty) {
		res = DataObjectFactory::create<DenseMatrix<VT>>(0, 0, false);
		goto exit;

	    // Dense Matrix
	    } else if (bb.bt == DF_body_t::dense) {
		 f.read((char *)&vt, sizeof(vt));

		 VT* memblock = (VT*) malloc(bb.nbrows*bb.nbcols*sizeof(VT));
		 f.read((char *)memblock, bb.nbrows*bb.nbcols*sizeof(VT));

		 std::shared_ptr<VT[]> data;
		 data.reset(memblock);

		 res = DataObjectFactory::create<DenseMatrix<VT>>((size_t)bb.nbrows, 
								  (size_t)bb.nbcols, data);

		 goto exit;
	    }
    }
exit:
   f.close();
   return;
  }
};

template <typename VT> struct ReadDaphne<CSRMatrix<VT>> {
  static void apply(CSRMatrix<VT> *&res, const char *filename) {

    std::ifstream f;
    f.open(filename, std::ios::in|std::ios::binary);
    // TODO: check f.good()

    // read header
    DF_header h;
    f.read((char *)&h, sizeof(h));

    if (h.dt == DF_data_t::CSRMatrix_t) {
	    uint8_t vt;
	    f.read((char *)&vt, sizeof(vt));

	    DF_body b;
	    f.read((char *)&b, sizeof(b));
	    // b is ignored for now - assumed to be 0,0
	    //TODO: consider multiple blocks

	    DF_body_block bb;
	    f.read((char *)&bb,sizeof(bb));
	    // empty Matrix
	    if (bb.bt == DF_body_t::empty) {
		res = DataObjectFactory::create<CSRMatrix<VT>>(0, 0, 0, false);
		goto exit;
	    // CSR Matrix
	    } else if (bb.bt == DF_body_t::sparse) {
		    uint8_t vt;
		    f.read((char *)&vt, sizeof(vt));

		    uint64_t nzb;
		    f.read((char *)&nzb, sizeof(nzb));

		    res = DataObjectFactory::create<CSRMatrix<VT>>(
				bb.nbrows, bb.nbcols, nzb, true);

		    for (size_t i = 0; i < bb.nbrows; i++) {
			    uint64_t nzr;
			    f.read((char *)&nzr, sizeof(nzr));

			    for (uint64_t n = 0; n < nzr; n++) {
				  size_t j;
				  f.read((char *)&j, sizeof(j));

				  VT val;
				  f.read((char *)&val, sizeof(val));

				  res->set(i, j, val);
			    }
		    }

		goto exit;

            // COO Matrix
	    } else if (bb.bt == DF_body_t::ultra_sparse) {
		    uint8_t vt;
		    f.read((char *)&vt, sizeof(vt));

		    uint64_t nzb;
		    f.read((char *)&nzb, sizeof(nzb));

		    res = DataObjectFactory::create<CSRMatrix<VT>>(
				bb.nbrows, bb.nbcols, nzb, false);

		   // Single column case
		   if (bb.nbcols == 1) {
			for (uint64_t n = 0; n < nzb; n++) {
				uint32_t i;
				f.read((char *)&i, sizeof(i));

				VT val;
				f.read((char *)&val, sizeof(val));

				res->set(i, 1, val);
			}
			goto exit;
		   } else {
			   // TODO: check numcols is greater than 1
			for (uint64_t n = 0; n < nzb; n++) {
				uint32_t i;
				f.read((char *)&i, sizeof(i));

				uint32_t j;
				f.read((char *)&j, sizeof(j));

				VT val;
				f.read((char *)&val, sizeof(val));

				res->set(i, j, val);
			}
			goto exit;
		   }

	    }
	    //TODO: frames
exit:
	    f.close();
	    return;
    }
  }
};


template <> struct ReadDaphne<Frame> {
  static void apply(Frame *&res, const char *filename){

    std::ifstream f;
    f.open(filename, std::ios::in|std::ios::binary);
    // TODO: check f.good()

    // read commong part of the header
    DF_header h;
    f.read((char *)&h, sizeof(h));

    if (h.dt == DF_data_t::Frame_t) {
	    // read rest of the header
	    ValueTypeCode *schema = new ValueTypeCode[h.nbcols];
	    for (uint64_t c = 0; c < h.nbcols; c++) {
		f.read((char *)&(schema[c]), sizeof(ValueTypeCode));
	    }

	    std::string *labels = new std::string[h.nbcols];
	    for (uint64_t c = 0; c < h.nbcols; c++) {
		uint16_t len;
		f.read((char *) &len, sizeof(len));
		f.read((char *) &(labels[c]), len);
	    }

	    res = DataObjectFactory::create<Frame>(
		h.nbrows, h.nbcols, schema,  
		types, true);

	    DF_body b;
	    f.read((char *)&b, sizeof(b));
	    // b is ignored for now - assumed to be 0,0
	    //TODO: consider multiple blocks
	    // Assuming a dense block representation
	    // TODO: Consider alternative representations for frames
	    for (uint64_t r = 0; c < h.nbrows; r++) {
		for (uint64_t c = 0; c < h.nbcols; c++) {
			char * val = new(char[ValueTypeUtils::sizeOf(schema[c])]);
			f.read((char *)val, ValueTypeUtils::sizeOf(schema[c]));
			(res->columns[c][r]).reset(val);
		}
	    }
    }
    f.close();
    return;
  }
};

#endif
