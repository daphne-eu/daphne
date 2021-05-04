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

#include <string.h>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, typename VT>
struct Replace {
	static void apply(DTRes *& res, DTArg *& arg, VT pattern, VT replacement) = delete;
};


// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg, typename VT>
void replace(DTRes *& res, DTArg *& arg, VT pattern, VT replacement ) {
	Replace<DTRes, DTArg, VT>::apply(res, arg, pattern, replacement);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
//
template<typename VT>
struct Replace<DenseMatrix<VT>, DenseMatrix<VT>, VT> {
	static void apply(DenseMatrix<VT> *& res, DenseMatrix<VT> *& arg, VT pattern, VT replacement) {
		//------handling corner cases -------
		if(arg==nullptr){//one might throw an exception - nullptr
			return;
		}
		const size_t numRows = arg->getNumRows(); // number of rows
		const size_t numCols = arg->getNumCols(); // number of columns
		const size_t rowSkipRes = arg->getRowSkip(); // number of row skips
		DenseMatrix<VT> *  targetMatrix = arg; // this will point to the matrix that should be updated. Default is arg "in-place update"
		bool requireCopy=false; // this variable is to indicate whether we need to copy to res (when not using inplace update semantic)
		if(numRows==0 && numRows==numCols){// one might throw an exception - empty matrix
			return;
		}
		if(res!=arg && res!=nullptr){ //one might throw an exception -- res should be either pointing to the same location of arg or equals to a nullptr
			return;
		}
		if((replacement!=replacement && pattern!=pattern) || (pattern == replacement)){// nothing to be done pattern equals replacement
			if(res!=nullptr){// do nothing
			}
			else{
				res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols,  false);
				//copy and return in this case replace will be a copy function that copies arg to res
				memcpy(res, arg, numRows*numRows*sizeof(VT));
				return;
			}
		}

		if(res==nullptr){
			res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
			requireCopy=true;
			targetMatrix=res;
		}

		//--------main logic --------------------------
		if(pattern!=pattern){ // pattern is NaN
			for(size_t r=0;r<numRows;r++){
				VT * valuesOfRow = arg->getValues() + (r*rowSkipRes);
				VT * updatedValuesOfRow = targetMatrix->getValues() + (r*rowSkipRes);
				for (size_t c=0; c<numCols;c++){
					if(valuesOfRow[c]!=valuesOfRow[c]){
						updatedValuesOfRow[c]=replacement;
					}
					else if(requireCopy){
						updatedValuesOfRow[c]=valuesOfRow[c];
					}
				}
			}
		}
		else{ // pattern is not NaN --> replacement can still be NaN
			for(size_t r=0;r<numRows;r++){
				VT * valuesOfRow = arg->getValues() + (r*rowSkipRes);
				VT * updatedValuesOfRow = targetMatrix->getValues() + (r*rowSkipRes);
				for (size_t c=0; c<numCols;c++){
					if(valuesOfRow[c]==pattern){
						updatedValuesOfRow[c]=replacement;
					}
					else if(requireCopy){
						updatedValuesOfRow[c]=valuesOfRow[c];
					}
				}
			}
		}
	}
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Replace<CSRMatrix<VT>, CSRMatrix<VT>, VT> {
	static void apply(CSRMatrix<VT> *& res, CSRMatrix<VT> *& arg, VT pattern, VT replacement) {
		//TODO
	}
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_REPLACE_H
