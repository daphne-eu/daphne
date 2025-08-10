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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct SaveDaphneLibResult {
    static void apply(const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> void saveDaphneLibResult(const DTArg *arg, DCTX(ctx)) {
    SaveDaphneLibResult<DTArg>::apply(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct SaveDaphneLibResult<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, DCTX(ctx)) {
        // Increase the reference counter of the data object to be transferred
        // to numpy, such that the data is not garbage collected by DAPHNE.
        // TODO But who will free the memory in the end?
        arg->increaseRefCounter();

        DaphneLibResult *daphneLibRes = ctx->getUserConfig().result_struct;

        if (!daphneLibRes)
            throw std::runtime_error("saveDaphneLibRes(): daphneLibRes is nullptr");

        daphneLibRes->address = const_cast<void *>(reinterpret_cast<const void *>(arg->getValues()));
        daphneLibRes->cols = arg->getNumCols();
        daphneLibRes->rows = arg->getNumRows();
        daphneLibRes->vtc = (int64_t)ValueTypeUtils::codeFor<VT>;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct SaveDaphneLibResult<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, DCTX(ctx)) {

        const_cast<CSRMatrix<VT> *>(arg)->increaseRefCounter();

        DaphneLibResult *daphneLibRes = ctx->getUserConfig().result_struct;
        if (!daphneLibRes)
            throw std::runtime_error("saveDaphneLibRes(): daphneLibRes is nullptr");

        // Result is a sparse matrix
        daphneLibRes->isSparse = true;

        const size_t rows = arg->getNumRows();
        const size_t cols = arg->getNumCols();
        daphneLibRes->rows = rows;
        daphneLibRes->cols = cols;
        daphneLibRes->vtc = (int64_t)ValueTypeUtils::codeFor<VT>;

        // original raw pointers
        auto origVals = arg->getValuesSharedPtr().get();
        auto origCols = arg->getColIdxsSharedPtr().get();
        auto origOffsets = arg->getRowOffsetsSharedPtr().get();

        // first pass: count *actual* non-zeros
        size_t actualNNZ = 0;
        for (size_t r = 0; r < rows; r++) {
            auto start = origOffsets[r];
            auto end = origOffsets[r + 1];
            for (size_t i = start; i < end; i++)
                if (origVals[i] != VT(0))
                    actualNNZ++;
        }

        // allocate new tight buffers
        VT *cleanVals = new VT[actualNNZ];
        size_t *cleanCols = new size_t[actualNNZ];
        size_t *cleanOffs = new size_t[rows + 1];

        // second pass: copy only non-zeros
        size_t p = 0;
        cleanOffs[0] = 0;
        for (size_t r = 0; r < rows; r++) {
            auto start = origOffsets[r];
            auto end = origOffsets[r + 1];
            for (size_t i = start; i < end; i++) {
                VT v = origVals[i];
                if (v != VT(0)) {
                    cleanVals[p] = v;
                    cleanCols[p] = origCols[i];
                    p++;
                }
            }
            cleanOffs[r + 1] = p;
        }

        // hand them off in the result struct
        daphneLibRes->data = static_cast<void *>(cleanVals);
        daphneLibRes->col_related = static_cast<void *>(cleanCols);
        daphneLibRes->row_related = static_cast<void *>(cleanOffs);
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct SaveDaphneLibResult<Frame> {
    static void apply(const Frame *arg, DCTX(ctx)) {
        // Increase the reference counter of the data object to be transferred
        // to python, such that the data is not garbage collected by DAPHNE.
        // TODO But who will free the memory in the end?
        arg->increaseRefCounter();

        DaphneLibResult *daphneLibRes = ctx->getUserConfig().result_struct;

        if (!daphneLibRes)
            throw std::runtime_error("saveDaphneLibRes(): daphneLibRes is nullptr");

        const size_t numCols = arg->getNumCols();

        // Create fresh arrays for vtcs, labels and columns.
        int64_t *vtcs = new int64_t[numCols];
        char **labels = new char *[numCols];
        void **columns = new void *[numCols];
        for (size_t i = 0; i < numCols; i++) {
            vtcs[i] = static_cast<int64_t>(arg->getSchema()[i]);
            labels[i] = const_cast<char *>(arg->getLabels()[i].c_str());
            columns[i] = const_cast<void *>(reinterpret_cast<const void *>(arg->getColumnRaw(i)));
        }

        daphneLibRes->cols = numCols;
        daphneLibRes->rows = arg->getNumRows();
        daphneLibRes->vtcs = vtcs;
        daphneLibRes->labels = labels;
        daphneLibRes->columns = columns;
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_SAVEDAPHNELIBRESULT_H
