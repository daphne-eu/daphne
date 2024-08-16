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

#include <cblas.h>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Syrk {
    static void apply(DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void syrk(DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    Syrk<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct Syrk<DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(numCols, numCols, false);

        cblas_dsyrk(CblasRowMajor,
            CblasUpper,
            CblasTrans,
            numCols,
            numRows,
            1.0,
            arg->getValues(),
            arg->getRowSkip(),
            0.0,
            res->getValues(),
            res->getRowSkip());
        for (auto r = 0u; r < numCols; ++r) {
            for (auto c = r + 1; c < numCols; ++c) {
                res->set(c, r, res->get(r, c));
            }
        }
    }
};

template<>
struct Syrk<DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(numCols, numCols, false);

        cblas_ssyrk(CblasRowMajor,
            CblasUpper,
            CblasTrans,
            numCols,
            numRows,
            1.0,
            arg->getValues(),
            arg->getRowSkip(),
            0.0,
            res->getValues(),
            res->getRowSkip());
        for (auto r = 0u; r < numCols; ++r) {
            for (auto c = r + 1; c < numCols; ++c) {
                res->set(c, r, res->get(r, c));
            }
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Syrk<CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numCols, numRows, arg->getNumNonZeros(), false);
        throw std::runtime_error("TODO: Syrk for Sparse");
    }
};
