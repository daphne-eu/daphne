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
#include <vector>
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

         size_t numItems = bb.nbrows*bb.nbcols;
         std::streamsize memBlockSize = numItems * sizeof(VT);
         auto memblock = std::shared_ptr<VT[]>(new VT[numItems], std::default_delete<VT[]>());
         f.read(reinterpret_cast<char*>(memblock.get()), memBlockSize);
		 res = DataObjectFactory::create<DenseMatrix<VT>>(static_cast<size_t>(bb.nbrows), static_cast<size_t>(bb.nbcols),
                 memblock);

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
	    ValueTypeCode * schema = new ValueTypeCode[h.nbcols];
	    for (uint64_t c = 0; c < h.nbcols; c++) {
		f.read((char *)&(schema[c]), sizeof(ValueTypeCode));
	    }

	    std::string *labels = new std::string[h.nbcols];
	    for (uint64_t c = 0; c < h.nbcols; c++) {
		uint16_t len;
		f.read((char *) &len, sizeof(len));
		f.read((char *) &(labels[c]), len);
	    }

	    DF_body b;
	    f.read((char *)&b, sizeof(b));
	    // b is ignored for now - assumed to be 0,0
	    //TODO: consider multiple blocks
	    // Assuming a dense block representation
	    // TODO: Consider alternative representations for frames

	    if (res == nullptr) {
		res = DataObjectFactory::create<Frame>(h.nbrows, h.nbcols, schema, nullptr, false);
	    }

	    uint8_t ** rawCols = new uint8_t * [h.nbcols];
            for(size_t i = 0; i < h.nbcols; i++) {
                rawCols[i] = reinterpret_cast<uint8_t *>(res->getColumnRaw(i));
            }

	    for (size_t r=0; r < h.nbrows; r++) {
			for (size_t c=0; c < h.nbcols; c++) { 
				switch (schema[c]) {
					case ValueTypeCode::SI8:
						int8_t val_si8;
						f.read((char *) &val_si8, sizeof(val_si8));
						reinterpret_cast<int8_t *>(rawCols[c])[r] = val_si8;
						break;
					case ValueTypeCode::SI32:
						int32_t val_si32;
						f.read((char *) &val_si32, sizeof(val_si32));
						reinterpret_cast<int32_t *>(rawCols[c])[r] = val_si32;
						break;
					case ValueTypeCode::SI64:
						int64_t val_si64;
						f.read((char *) &val_si64, sizeof(val_si64));
						reinterpret_cast<int64_t *>(rawCols[c])[r] = val_si64;
						break;
					case ValueTypeCode::UI8:
						uint8_t val_ui8;
						f.read((char *) &val_ui8, sizeof(val_ui8));
						reinterpret_cast<uint8_t *>(rawCols[c])[r] = val_ui8;
						break;
					case ValueTypeCode::UI32:
						uint32_t val_ui32;
						f.read((char *) &val_ui32, sizeof(val_ui32));
						reinterpret_cast<uint32_t *>(rawCols[c])[r] = val_ui32;
						break;
					case ValueTypeCode::UI64:
						uint64_t val_ui64;
						f.read((char *) &val_ui64, sizeof(val_ui64));
						reinterpret_cast<uint64_t *>(rawCols[c])[r] = val_ui64;
						break;
					case ValueTypeCode::F32:
						float val_f32;
						f.read((char *) &val_f32, sizeof(val_f32));
						reinterpret_cast<float *>(rawCols[c])[r] = val_f32;
						break;
					case ValueTypeCode::F64:
						double val_f64;
						f.read((char *) &val_f64, sizeof(val_f64));
						reinterpret_cast<double *>(rawCols[c])[r] = val_f64;
						break;
					default:
						throw std::runtime_error("ReadDaphne::apply: unknown value type code");
				}
			}
        }

	delete[] rawCols;
	delete[] schema;
    }
    f.close();
    return;
  }
};
