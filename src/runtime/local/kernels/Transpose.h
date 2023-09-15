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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/InPlaceUtils.h>
#include <spdlog/spdlog.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Transpose {
    static void apply(DTRes *& res, DTArg * arg, bool hasFutureUseArg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void transpose(DTRes *& res, DTArg * arg, bool hasFutureUseArg, DCTX(ctx)) {
    Transpose<DTRes, DTArg>::apply(res, arg, hasFutureUseArg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Transpose<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, DenseMatrix<VT> * arg, bool hasFutureUseArg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        // skip data movement for vectors
        // FIXME: The check (numCols == arg->getRowSkip()) is a hack to check if the input arg is only a "view"
        //        on a larger matrix.

        if ((numRows == 1 || numCols == 1) && (numCols == arg->getRowSkip())) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows, arg);
        }
        else
        {

            if (res == nullptr) {
                if(InPlaceUtils::isInPlaceable(arg, hasFutureUseArg)) {
                    // In case of square matrix, we can transpose in-place on the data object
                    if(numRows == numCols) {
                        spdlog::debug("Transpose(Dense) - arg is in-placeable");
                        res = arg;
                        res->increaseRefCounter();
                    }
                    // In case of non-square matrix, we need to allocate a new object
                    // but we can still transpose in-place on the data buffer.
                    else {
                        spdlog::debug("Transpose(Dense) - data buffer of arg is in-placeable");
                        res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows, arg);
                    }

                    // We need to apply a different algorithm for updating in-place matrices
                    // Based on https://en.wikipedia.org/wiki/In-place_matrix_transposition#Non-square_matrices%3a_Following_the_cycles
                    // and https://www.geeksforgeeks.org/inplace-m-x-n-size-matrix-transpose/
                    // Here we initialize a boolean array to keep track of the visited elements.
                    // Trade-off between allocating n*m bytes (boolean) vs worst case n*m*8 bytes (double)
                    // TODO: Implement trivial algorithm for square matrices

                    VT *valuesArg = arg->getValues();
                    int size = numRows * numCols;
                    bool* visited = new bool[size]{false};

                    for (int start = 0; start < size; ++start) {

                        if (visited[start])
                            continue;

                        VT temp = valuesArg[start];
                        int current = start;

                        do {
                            int next = (current % numCols) * numRows + current / numCols;
                            VT nextValue = valuesArg[next];
                            valuesArg[next] = temp;
                            visited[current] = true;
                            current = next;
                            temp = nextValue;
                        } while (current != start);
                    }

                    delete[] visited;

                    //early return
                    return;

                }
                else {
                    spdlog::debug("Transpose(Dense) - create new matrix for result");
                    res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows, false);
                }
            }

            // Default case
            const VT *valuesArg = arg->getValues();
            const size_t rowSkipArg = arg->getRowSkip();
            const size_t rowSkipRes = res->getRowSkip();
            for (size_t r = 0; r < numRows; r++) {
                VT *valuesRes = res->getValues() + r;
                for (size_t c = 0; c < numCols; c++) {
                    *valuesRes = valuesArg[c];
                    valuesRes += rowSkipRes;
                }
                valuesArg += rowSkipArg;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Transpose<CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, CSRMatrix<VT> * arg, bool hasFutureUseArg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(hasFutureUseArg == false) {
            spdlog::debug("Transpose(CSR) - in-place transpose of CSRMatrix is not supported");
        }
        
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numCols, numRows, arg->getNumNonZeros(), false);
        
        const VT * valuesArg = arg->getValues();
        const size_t * colIdxsArg = arg->getColIdxs();
        const size_t * rowOffsetsArg = arg->getRowOffsets();
        
        VT * valuesRes = res->getValues();
        VT * const valuesResInit = valuesRes;
        size_t * colIdxsRes = res->getColIdxs();
        size_t * rowOffsetsRes = res->getRowOffsets();
        
        auto* curRowOffsets = new size_t[numRows + 1];
        memcpy(curRowOffsets, rowOffsetsArg, (numRows + 1) * sizeof(size_t));
        
        rowOffsetsRes[0] = 0;
        for(size_t c = 0; c < numCols; c++) {
            for(size_t r = 0; r < numRows; r++)
                if(curRowOffsets[r] < rowOffsetsArg[r + 1] && colIdxsArg[curRowOffsets[r]] == c) {
                    *valuesRes++ = valuesArg[curRowOffsets[r]];
                    *colIdxsRes++ = r;
                    curRowOffsets[r]++;
                }
            rowOffsetsRes[c + 1] = valuesRes - valuesResInit;
        }
        
        delete[] curRowOffsets;
    }
};
