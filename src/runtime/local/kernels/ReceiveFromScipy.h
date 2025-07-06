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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMSCIPY_H
#define SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMSCIPY_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <memory>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template <class DTRes> struct ReceiveFromScipy {
    static void apply(DTRes *&res, uint64_t valuesAddr, uint64_t rowRelatedIdx, uint64_t colRelatedIdx, int64_t numRows,
                      int64_t numCols, int64_t nnz, uint64_t format, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void receiveFromScipy(DTRes *&res, uint64_t valuesAddr, uint64_t rowRelatedIdx, uint64_t colRelatedIdx, int64_t numRows,
                      int64_t numCols, int64_t nnz, uint64_t format, DCTX(ctx)) {
    ReceiveFromScipy<DTRes>::apply(res, valuesAddr, rowRelatedIdx, colRelatedIdx, numRows, numCols, nnz, format, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// TODO Should we make this a central utility?
template <typename VT> struct NoOpDeleter {
    void operator()(VT *p) {
        // Don't delete p because the memory comes from numpy.
    }
};

template <typename VT> struct ReceiveFromScipy<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, uint64_t valuesAddr, uint64_t rowRelatedIdx, uint64_t colRelatedIdx,
                      int64_t numRows, int64_t numCols, int64_t nnz, uint64_t format, DCTX(ctx)) {
        std::shared_ptr<VT[]> values((VT *)valuesAddr, NoOpDeleter<VT>());
        std::shared_ptr<size_t[]> colIdxs;
        std::shared_ptr<size_t[]> rowOffsets;

        if (format == 0) { // CSR
            auto rowOffsetsRaw = (size_t *)rowRelatedIdx;
            auto colIdxsRaw = (size_t *)colRelatedIdx;

            rowOffsets = std::shared_ptr<size_t[]>(rowOffsetsRaw, NoOpDeleter<size_t>());
            colIdxs = std::shared_ptr<size_t[]>(colIdxsRaw, NoOpDeleter<size_t>());
        } else if (format == 1) { // COO
            auto rowIdxsRaw = (size_t *)rowRelatedIdx;
            auto colIdxsRaw = (size_t *)colRelatedIdx;

            colIdxs = std::shared_ptr<size_t[]>(colIdxsRaw, NoOpDeleter<size_t>());

            size_t *rowOffsetsRaw = new size_t[numRows + 1]();

            for (int64_t i = 0; i < nnz; i++) {
                size_t r = rowIdxsRaw[i];
                if (r >= (size_t)numRows)
                    throw std::runtime_error("ReceiveFromScipy: failed to convert COO input to CSR.");
                rowOffsetsRaw[r + 1]++;
            }

            for (int64_t i = 0; i < numRows; i++) {
                rowOffsetsRaw[i + 1] += rowOffsetsRaw[i];
            }

            colIdxs = std::shared_ptr<size_t[]>(colIdxsRaw, NoOpDeleter<size_t>());
            rowOffsets = std::shared_ptr<size_t[]>(rowOffsetsRaw, std::default_delete<size_t[]>());

        } else if (format == 2) { // CSC
            auto rowIdxsRaw = (size_t *)rowRelatedIdx;
            auto colOffsetsRaw = (size_t *)colRelatedIdx;

            // Build CSR row-offsets
            size_t *rowOffsetsRaw = new size_t[numRows + 1]();
            for (int64_t col = 0; col < numCols; col++) {
                size_t start = colOffsetsRaw[col], end = colOffsetsRaw[col + 1];
                for (size_t i = start; i < end; i++) {
                    size_t r = rowIdxsRaw[i];
                    if (r >= (size_t)numRows)
                        throw std::runtime_error("ReceiveFromScipy: failed to convert CSC input to CSR.");
                    rowOffsetsRaw[r + 1]++;
                }
            }
            for (int64_t i = 0; i < numRows; i++)
                rowOffsetsRaw[i + 1] += rowOffsetsRaw[i];

            // Allocate CSR storage column-indices and values.
            size_t *colIdxsRaw = new size_t[nnz];
            VT *dataCsrRaw = new VT[nnz];

            // Scatter into CSR, tracking per-row insertion
            std::vector<size_t> nextInsert(numRows);
            for (int64_t i = 0; i < numRows; i++)
                nextInsert[i] = rowOffsetsRaw[i];

            for (int64_t col = 0; col < numCols; col++) {
                size_t start = colOffsetsRaw[col], end = colOffsetsRaw[col + 1];
                for (size_t i = start; i < end; i++) {
                    size_t r = rowIdxsRaw[i];
                    size_t pos = nextInsert[r]++;
                    colIdxsRaw[pos] = col;
                    dataCsrRaw[pos] = values.get()[i];
                }
            }

            // Wrap into shared_ptrs
            colIdxs = std::shared_ptr<size_t[]>(colIdxsRaw, std::default_delete<size_t[]>());
            rowOffsets = std::shared_ptr<size_t[]>(rowOffsetsRaw, std::default_delete<size_t[]>());
            values = std::shared_ptr<VT[]>(dataCsrRaw, std::default_delete<VT[]>());
        } else {
            throw std::runtime_error("ReceiveFromScipy: Unknown sparse matrix representation.");
        }

        res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, nnz, values, colIdxs, rowOffsets);
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMSCIPY_H
