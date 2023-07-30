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

#endif //SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H
