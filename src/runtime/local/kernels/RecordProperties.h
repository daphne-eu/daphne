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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_LOG_PROPERTIES_H
#define SRC_RUNTIME_LOCAL_KERNELS_LOG_PROPERTIES_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/SparseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <variant>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct RecordProperties {
    static void apply(DTRes *& res, const DTArg * arg, const std::string &op_id, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void recordProperties(DTRes *& res, const DTArg * arg, const std::string &op_id, DCTX(ctx)) {
    RecordProperties<DTRes, DTArg>::apply(res, arg, op_id, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix Record Implementation
// ----------------------------------------------------------------------------

template<typename VT>
struct RecordProperties<void, DenseMatrix<VT>> {
    static void apply(void *& res, const DenseMatrix<VT> * arg, const std::string &op_id, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t nnz = arg->getNumNonZeros();
        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);

        ctx->propertyLogger.logProperty(op_id, "shape", std::make_pair(numRows, numCols));
        ctx->propertyLogger.logProperty(op_id, "cardinality", numRows * numCols);
        ctx->propertyLogger.logProperty(op_id, "type", "DenseMatrix<" + std::string(typeid(VT).name()) + ">");
        ctx->propertyLogger.logProperty(op_id, "sparsity", sparsity);
    }
};

// ----------------------------------------------------------------------------
// SparseMatrix Record Implementation
// ----------------------------------------------------------------------------

template<typename VT>
struct RecordProperties<void, SparseMatrix<VT>> {
    static void apply(void *& res, const SparseMatrix<VT> * arg, const std::string &op_id, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t nnz = arg->getNumNonZeros();
        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);

        ctx->propertyLogger.logProperty(op_id, "shape", std::make_pair(numRows, numCols));
        ctx->propertyLogger.logProperty(op_id, "cardinality", numRows * numCols);
        ctx->propertyLogger.logProperty(op_id, "type", "SparseMatrix<" + std::string(typeid(VT).name()) + ">");
        ctx->propertyLogger.logProperty(op_id, "sparsity", sparsity);
    }
};

// ----------------------------------------------------------------------------
// General DataObject Record Implementation
// ----------------------------------------------------------------------------

template<typename DTArg>
struct RecordProperties<void, DTArg> {
    static void apply(void *& res, const DTArg * arg, const std::string &op_id, DCTX(ctx)) {
        ctx->propertyLogger.logProperty(op_id, "type", typeid(DTArg).name());
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_LOG_PROPERTIES_H
