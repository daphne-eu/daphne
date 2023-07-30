/*
 * Copyright 2022 The DAPHNE Consortium
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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H
#define SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTArg>
struct SaveDaphneLibResult {
    static void apply(const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTArg>
void saveDaphneLibResult(const DTArg * arg, DCTX(ctx)) {
    SaveDaphneLibResult<DTArg>::apply(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct SaveDaphneLibResult<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> * arg,  DCTX(ctx)) {
        // Increase the reference counter of the data object to be transferred
        // to numpy, such that the data is not garbage collected by DAPHNE.
        // TODO But who will free the memory in the end?
        arg->increaseRefCounter();

        DaphneLibResult* daphneLibRes = ctx->getUserConfig().result_struct;
        
        if(!daphneLibRes)
            throw std::runtime_error("saveDaphneLibRes(): daphneLibRes is nullptr");

        daphneLibRes->address = const_cast<void*>(reinterpret_cast<const void*>(arg->getValues()));
        daphneLibRes->cols = arg->getNumCols();
        daphneLibRes->rows = arg->getNumRows();
        daphneLibRes->vtc = (int64_t)ValueTypeUtils::codeFor<VT>;
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template<>
struct SaveDaphneLibResult<Frame> {
    static void apply(const Frame * arg,  DCTX(ctx)) {
        // Increase the reference counter of the data object to be transferred
        // to python, such that the data is not garbage collected by DAPHNE.
        // TODO But who will free the memory in the end?
        // Memory allocated with new has to be freed manually with delete[]. 
        // Therefore, we should call delete[] for vtcs and each element of labels array 
        // (and then for labels array itself) when you're done with these arrays.
        // Delete Function for Both DenseMatrix and Frame should be implemented.

        arg->increaseRefCounter();

        DaphneLibResult* daphneLibRes = ctx->getUserConfig().result_struct;
        
        if(!daphneLibRes)
            throw std::runtime_error("saveDaphneLibRes(): daphneLibRes is nullptr");

        std::vector<ValueTypeCode> vtcs_tmp;    // The tmp arrays do not need to be deleted manually
        std::vector<std::string> labels_tmp;    // They are local dynamic arrays and the destructor is called automatically

        for(size_t i = 0; i < arg->getNumCols(); i++) {
            vtcs_tmp.push_back(arg->getSchema()[i]);
            labels_tmp.push_back(arg->getLabels()[i]);
        }

        // Create C-Type arrays for vtcs, labels and columns
        int64_t* vtcs = new int64_t[arg->getNumCols()];
        char** labels = new char*[arg->getNumCols()];
        void** columns = new void*[arg->getNumCols()];

        // Assign the Frame Information to the C-Type Arrays
        for(size_t i = 0; i < arg->getNumCols(); i++) {
            vtcs[i] = static_cast<int64_t>(vtcs_tmp[i]);
            labels[i] = new char[labels_tmp[i].size() + 1];
            strcpy(labels[i], labels_tmp[i].c_str());

            columns[i] = const_cast<void*>(reinterpret_cast<const void*>(arg->getColumnRaw(i)));
        }

        daphneLibRes->cols = arg->getNumCols();
        daphneLibRes->rows = arg->getNumRows();
        daphneLibRes->vtcs = vtcs;
        daphneLibRes->labels = labels;
        daphneLibRes->columns = columns; 
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H
