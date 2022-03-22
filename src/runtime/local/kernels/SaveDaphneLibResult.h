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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H
#define SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/FileMetaData.h>
#include "runtime/local/io/DaphneLibResult.h"

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
      arg->increaseRefCounter();
      daphne_lib_res.address = (void*)arg->getValues();
      daphne_lib_res.cols = arg->getNumCols();
      daphne_lib_res.rows = arg->getNumRows();
      daphne_lib_res.vtc = (int)ValueTypeUtils::codeFor<VT>;
    }
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H