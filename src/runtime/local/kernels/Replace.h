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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_REPLACE_H
#define SRC_RUNTIME_LOCAL_KERNELS_REPLACE_H

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VT>
struct Replace {
    static void apply(DTRes *& res, VT pattern, VT replacement) = delete;
};


// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, typename VT>
void replace(DTRes *& res, VT pattern, VT replacement ) {
    Replace<DTRes, VT>::apply(res, pattern, replacement);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// 
template<typename VT>
struct Replace<DenseMatrix<VT>, VT> {
	static void apply(DenseMatrix<VT> *& res, VT pattern, VT replacement) {
		//------handling corner cases -------
		if(res==nullptr){//one might throw an exception - nullptr 
			return;
		}
	        const size_t numRows = res->getNumRows();
        	const size_t numCols = res->getNumCols();
		const size_t rowSkipRes = res->getRowSkip();
		if(numRows==0 && numRows==numCols){// one might throw an exception - empty matrix 
			return;	
		}
		if((replacement!=replacement && pattern!=pattern) || (pattern == replacement)){// nothing to be done pattern equals replacement
			return;
		}
		//--------main logic -------------------------- 
		if(pattern!=pattern){ // pattern is NaN
			for(size_t r=0;r<numRows;r++){
				VT * valuesOfRow = res->getValues() + (r*rowSkipRes);
				for (size_t c=0; c<numCols;c++){
					if(valuesOfRow[c]!=valuesOfRow[c])
						valuesOfRow[c]=replacement;
				}
			}
		}
		else{ // pattern is not NaN --> replacement can still be NaN
			 for(size_t r=0;r<numRows;r++){
                                VT * valuesOfRow = res->getValues() + (r*rowSkipRes);
                                for (size_t c=0; c<numCols;c++){
                                        if(valuesOfRow[c]==pattern)
                                                valuesOfRow[c]=replacement;
                                }
                        }	
		}
	}
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Replace<CSRMatrix<VT>, VT> {
        static void apply(CSRMatrix<VT> *& res, VT pattern, VT replacement) {
	//TODO 
	}
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_REPLACE_H
