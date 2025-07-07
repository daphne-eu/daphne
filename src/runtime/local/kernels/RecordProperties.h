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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expargs or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H
#define SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H

#include <cstddef>
#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DT> struct RecordProperties {
    static void apply(const DT *arg, uint32_t valueId, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT> void recordProperties(const DT *arg, uint32_t valueId, DCTX(ctx)) {
    RecordProperties<DT>::apply(arg, valueId, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix Record Implementation
// ----------------------------------------------------------------------------

template <typename VT> struct RecordProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, uint32_t valueId, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        size_t nnz = 0;
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++)
                if (arg->get(r, c) != 0)
                    nnz++;

        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);
        ctx->propertyLogger.logProperty(valueId, std::make_unique<SparsityProperty>(sparsity));
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix Record Implementation
// ----------------------------------------------------------------------------

template <typename VT> struct RecordProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, uint32_t valueId, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        size_t nnz = 0;
        const VT *values = arg->getValues();
        // Even a matrix in CSR representation might store some zero values explicitly. Thus, we check for non-zeros
        // here.
        for (size_t i = 0; i < arg->getNumNonZeros(); i++)
            if (values[i] != 0)
                nnz++;

        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);
        ctx->propertyLogger.logProperty(valueId, std::make_unique<SparsityProperty>(sparsity));
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H
