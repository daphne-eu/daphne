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

#ifndef SRC_RUNTIME_LOCAL_IO_WRITEDAPHNE_H
#define SRC_RUNTIME_LOCAL_IO_WRITEDAPHNE_H

#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <runtime/local/io/utils.h>
#include <runtime/local/io/DaphneFile.h>

#include <type_traits>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <stdlib.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg>
struct WriteDaphne {
    static void apply(const DTArg *arg, char * filename) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg>
void writeDaphne(const DTArg *arg, char * filename) {
    WriteDaphne<DTArg>::apply(arg, filename);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct WriteDaphne<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, char * filename) {

	std::ofstream f;
	f.open(filename, std::ios::out|std::ios::binary);
	// TODO: check f.good()

	// write header
	DF_header h;
	h.version = 1;
	h.dt = DF_data_t::DenseMatrix_t;
	h.nbrows = (uint64_t) arg->getNumRows();
	h.nbcols = (uint64_t) arg->getNumCols();
	f.write((const char *)&h, sizeof(h));

	// value type
	const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;	
	f.write((const char *) &vt, sizeof(vt)); 

	// write body
        // single block
	DF_body b;
	b.rx = 0;
	b.cx = 0;
	f.write((const char *)&b, sizeof(b));

	// block header
	DF_body_block bb;
	bb.nbrows = (uint32_t) arg->getNumRows();
	bb.nbcols = (uint32_t) arg->getNumCols();
	bb.bt = DF_body_t::dense;
	f.write((const char *)&bb, sizeof(bb));

	// value type
	f.write((const char *) &vt, sizeof(vt)); 

	// block values
	const VT * valuesArg = arg->getValues();
	f.write((const char *)valuesArg, arg->getNumRows() * arg->getNumCols() * sizeof(VT));

	f.close();
	return;
   }
};
  
template <typename VT>
struct WriteDaphne<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, char * filename) {

	std::ofstream f;
	f.open(filename, std::ios::out|std::ios::binary);
	// TODO: check f.good()

	// write header
	DF_header h;
	h.version = 1;
	h.dt = DF_data_t::CSRMatrix_t;
	h.nbrows = (uint64_t) arg->getNumRows();
	h.nbcols = (uint64_t) arg->getNumCols();
	f.write((const char *)&h, sizeof(h));

	// value type
	const ValueTypeCode vt = ValueTypeUtils::codeFor<VT>;	
	f.write((const char *) &vt, sizeof(vt)); 

	// write body
        // single block
	DF_body b;
	b.rx = 0;
	b.cx = 0;
	f.write((const char *)&b, sizeof(b));

	// block header
	DF_body_block bb;
	bb.nbrows = (uint32_t) arg->getNumRows();
	bb.nbcols = (uint32_t) arg->getNumCols();
	bb.bt = DF_body_t::sparse;
	f.write((const char *)&bb, sizeof(bb));

	// value type
	f.write((const char *) &vt, sizeof(vt));

	// nzb value
        const size_t nzb = arg->getNumNonZeros();
	f.write((const char *) &nzb, sizeof(nzb) );

	// CSR values
	for (auto i = 0; i < arg->getNumRows(); i++) {
		// nzv for row i
		const size_t nzb_i = arg->getNumNonZeros(i);
	        f.write((const char *) &nzb_i, sizeof(nzb_i));

		// values for row i
		const VT * vals= arg->getValues(i);
		const size_t * colIdxs= arg->getColIdxs(i);

		for (auto j=0; j < nzb_i; j++) {
			f.write((const char *) &colIdxs[j], sizeof(colIdxs[j]));
			f.write((const char *) &(vals[j]), sizeof(VT));
		}
	}

	f.close();
	return;
   }
};
  
#endif // SRC_RUNTIME_LOCAL_IO_WRITECSV_H
