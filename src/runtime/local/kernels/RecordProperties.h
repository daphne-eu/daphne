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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H
#define SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H

#include <cstddef>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <string>
#include <typeinfo>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct RecordProperties {
    static void apply(const DT * res, const char* &op_id, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void recordProperties(const DT * res, const char* &op_id, DCTX(ctx)) {
    RecordProperties<DT>::apply(res, op_id, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix Record Implementation
// ----------------------------------------------------------------------------

template<typename VT>
struct RecordProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> * res, const char* &op_id, DCTX(ctx)) {
        const size_t numRows = res->getNumRows();
        const size_t numCols = res->getNumCols();
        size_t nnz = 0;
        
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                ++nnz;
            }
        }

        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);

        ctx->propertyLogger.logProperty(op_id, "shape", std::make_pair(numRows, numCols));
        ctx->propertyLogger.logProperty(op_id, "cardinality", numRows * numCols);
        ctx->propertyLogger.logProperty(op_id, "type", "DenseMatrix<" + std::string(typeid(VT).name()) + ">");
        ctx->propertyLogger.logProperty(op_id, "sparsity", sparsity);
    }
};

// ----------------------------------------------------------------------------
// CSR Record Implementation
// ----------------------------------------------------------------------------

template<typename VT>
struct RecordProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> * res, const char* &op_id, DCTX(ctx)) {
        const size_t numRows = res->getNumRows();
        const size_t numCols = res->getNumCols();

        const size_t nnz = res->getNumNonZeros();
        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);

        std::pair<size_t, size_t> shapes = {numRows, numCols};
        ctx->propertyLogger.logProperty(op_id, "shape", shapes);
        ctx->propertyLogger.logProperty(op_id, "cardinality", numRows * numCols);
        ctx->propertyLogger.logProperty(op_id, "type", "CSRMatrix<" + std::string(typeid(VT).name()) + ">");
        ctx->propertyLogger.logProperty(op_id, "sparsity", sparsity);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H
