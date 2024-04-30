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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <string.h>
#include <cstddef>
#include <stdio.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, typename VT>
struct Replace {
    static void apply(DTRes *& res, const DTArg * arg, VT pattern, VT replacement, DCTX(ctx)) = delete;
};


// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg, typename VT>
void replace(DTRes *& res, const DTArg * arg, VT pattern, VT replacement, DCTX(ctx)) {
    Replace<DTRes, DTArg, VT>::apply(res, arg, pattern, replacement, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Replace<DenseMatrix<VT>, DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, VT pattern, VT replacement, DCTX(ctx)) {
        //------handling corner cases -------
        if (!arg) {
            throw std::runtime_error(
                "Replace - arg must not be nullptr");
        }
        // variable declaration
        const size_t numRows = arg->getNumRows(); // number of rows
        const size_t numCols = arg->getNumCols(); // number of columns
        const size_t elementCount = numRows * numCols;
        bool requireCopy=false; // this variable is to indicate whether we need to copy to res (when not using inplace update semantic)s
        if(elementCount==0){// This case means that the kernel do nothing, i.e.,  no values to replace
            return;
        }
        if (res != nullptr) { // In this case, the caller reuses the res matrix
            if (res->getNumRows() != numRows) {
                throw std::runtime_error("res is a not a nullptr but it has a "
                                         "different numRows than arg");
            }
            if (res->getNumCols() != numCols) {
                throw std::runtime_error("res is a not a nullptr but it has a "
                                         "different numCols than arg");
            }
        }
        if((replacement!=replacement && pattern!=pattern) || (pattern == replacement)){// nothing to be done pattern equals replacement
            if(res!=nullptr && res==arg){  // arg and res are the same
                return; 
            }
            else if (res==nullptr){
                res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols,  false);
            }
            //copy and return in this case replace will be a copy function that copies arg to res
            const VT * allValues = arg->getValues();
            VT * allUpdatedValues = res->getValues();
            for(size_t r = 0; r < numRows; r++){
                for(size_t c = 0; c < numCols; c++){
                    allUpdatedValues[c]=allValues[c];
                }
                allUpdatedValues += res->getRowSkip();
                allValues += arg->getRowSkip();
            }
            return;
        }
        if(res==nullptr){
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
            requireCopy=true;
        }
        const VT * allValues = arg->getValues() ;
        VT * allUpdatedValues = res->getValues();
        //--------main logic --------------------------
        if(pattern!=pattern){ // pattern is NaN
            for(size_t r = 0; r < numRows; r++){
                for(size_t c = 0; c < numCols; c++){
                    if(allValues[c]!=allValues[c]){
                        allUpdatedValues[c]=replacement;
                    }
                    else if(requireCopy){
                        allUpdatedValues[c]=allValues[c];
                    }
                }
                allUpdatedValues += res->getRowSkip();
                allValues += arg->getRowSkip();
            }
        }
        else{ // pattern is not NaN --> replacement can still be NaN
            for(size_t r = 0; r < numRows; r++){
                for(size_t c = 0; c < numCols; c++){
                    if(allValues[c]==pattern){
                        allUpdatedValues[c]=replacement;
                    }
                    else if(requireCopy){
                        allUpdatedValues[c]=allValues[c];
                    }
                }
                allUpdatedValues += res->getRowSkip();
                allValues += arg->getRowSkip();    
            }
        }
    }   
};


// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Replace<CSRMatrix<VT>, CSRMatrix<VT>, VT> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * arg, VT pattern, VT replacement, DCTX(ctx)) {
        if (!arg) {
            throw std::runtime_error("Replace - arg must not be nullptr");
        }
        if (!pattern) {
            throw std::runtime_error("Replace - pattern equals zero, this case "
                                     "is not supported for now");
        }
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t nnzElements= arg->getNumNonZeros();
        if(nnzElements==0){// This case means that the kernel do nothing, i.e.,  no values to replace
            return;
        }
        if(res!=arg && res!=nullptr){ // In this case, the caller reuses the res matrix
            if (res->getNumRows() != numRows) {
                throw std::runtime_error("res is a not a nullptr but it has a "
                                         "different numRows than arg");
            }
            if (res->getNumCols() != numCols) {
                throw std::runtime_error("res is a not a nullptr but it has a "
                                         "different numCols than arg");
            }
            if (res->getNumNonZeros() != nnzElements) {
                throw std::runtime_error("res is a not a nullptr but it has a "
                                         "different nnzElements than arg");
            }
        }
        if((replacement!=replacement && pattern!=pattern) || (pattern == replacement)){// nothing to be done pattern equals replacement
            if(res!=nullptr && res==arg){  // arg and res are the same
                return; 
            }
            else if (res==nullptr){
                res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols,  nnzElements, false);
            }
            //copy and return in this case replace will be a copy function that copies arg to res
            memcpy(res->getRowOffsets(), arg->getRowOffsets(), (numRows+1)*sizeof(size_t));
            memcpy(res->getColIdxs(), arg->getColIdxs(), nnzElements*sizeof(size_t));
            memcpy(res->getValues(), arg->getValues(), nnzElements*sizeof(VT));
            return;
        }
        if(res==nullptr){
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols,  nnzElements, false);
            memcpy(res->getRowOffsets(), arg->getRowOffsets(), (numRows+1)*sizeof(size_t));
            memcpy(res->getColIdxs(), arg->getColIdxs(), nnzElements*sizeof(size_t));
            memcpy(res->getValues(), arg->getValues(), nnzElements*sizeof(VT));

        }
        //--------main logic --------------------------
        if(pattern!=pattern){ // pattern is NaN
            for(size_t r = 0; r < numRows; r++){
                const VT * allValues = arg->getValues(r);
                VT * allUpdatedValues = res->getValues(r);
                const size_t nnzElementsRes= arg->getNumNonZeros(r);
                for(size_t c = 0; c < nnzElementsRes; c++){
                    if(allValues[c]!=allValues[c]){
                        allUpdatedValues[c]=replacement;
                    }
                }
            }
        }
        else{
            for(size_t r = 0; r < numRows; r++){
                const VT * allValues = arg->getValues(r);
                VT * allUpdatedValues = res->getValues(r);
                const size_t nnzElementsRes= arg->getNumNonZeros(r);
                for(size_t c = 0; c < nnzElementsRes; c++){
                    if(allValues[c]==pattern){
                        allUpdatedValues[c]=replacement;
                    }
                }
            }
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_REPLACE_H
