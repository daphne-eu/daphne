#pragma once

#include "runtime/local/datastructures/CSRMatrix.h"
#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include <runtime/local/context/DaphneContext.h>

#include <algorithm>
#include <cmath>

template <typename DTRes, typename DTArg> struct Softmax {
    static void apply(DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

template <typename DTRes, typename DTArg> void softmax(DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    Softmax<DTRes, DTArg>::apply(res, arg, ctx);
}

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------
template <typename VT> struct Softmax<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        const VT *valuesArg = arg->getValues();
        VT *valuesRes = res->getValues();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        // Process each row independently
        for (size_t r = 0; r < numRows; r++) {
            // Compute max for numerical stability
            VT maxVal = valuesArg[0];
            for (size_t c = 1; c < numCols; c++)
                maxVal = std::max(maxVal, valuesArg[c]);

            // Compute exp and sum
            VT sum = 0;
            for (size_t c = 0; c < numCols; c++) {
                valuesRes[c] = std::exp(valuesArg[c] - maxVal);
                sum += valuesRes[c];
            }

            // Normalize
            const VT invSum = VT{1} / sum;
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] *= invSum;

            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes;
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------
template <typename VT> struct Softmax<DenseMatrix<VT>, CSRMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const CSRMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        VT *valuesRes = res->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        for (size_t r = 0; r < numRows; ++r) {
            auto nnz = arg->getNumNonZeros(r);
            const VT *rowValues = arg->getValues(r);
            const size_t *rowColIdxs = arg->getColIdxs(r);

            VT maxVal = nnz ? rowValues[0] : VT{0};
            if (nnz < numCols)
                maxVal = std::max(maxVal, VT{0});
            for (size_t i = 1; i < nnz; ++i)
                maxVal = std::max(maxVal, rowValues[i]);

            const VT zeroVal = std::exp(VT{0} - maxVal);
            std::fill_n(valuesRes, numCols, zeroVal);
            VT sum = zeroVal * static_cast<VT>(numCols);

            for (size_t i = 0; i < nnz; ++i) {
                const size_t colIdx = rowColIdxs[i];
                const VT val = std::exp(rowValues[i] - maxVal);
                sum += val - zeroVal;
                valuesRes[colIdx] = val;
            }

            const VT invSum = VT{1} / sum;
            for (size_t c = 0; c < numCols; ++c)
                valuesRes[c] *= invSum;

            valuesRes += rowSkipRes;
        }
    }
};
