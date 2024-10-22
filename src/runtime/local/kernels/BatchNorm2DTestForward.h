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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <limits>
#include <random>
#include <type_traits>

#include <cstddef>
#include <cstdint>

#include <cmath>
#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct BatchNorm2DTestForward {
    static void apply(DTRes *&res, const DTArg *in, const DTArg *gamma, const DTArg *beta, const DTArg *emaMean,
                      const DTArg *emaVar, const typename DTArg::VT eps, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void batchNorm2DTestForward(DTRes *&res, const DTArg *in, const DTArg *gamma, const DTArg *beta, const DTArg *emaMean,
                            const DTArg *emaVar, const typename DTArg::VT eps, DCTX(dctx)) {
    BatchNorm2DTestForward<DTRes, DTArg>::apply(res, in, gamma, beta, emaMean, emaVar, eps, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> static inline VT getMean(const VT *in, uint32_t start, uint32_t length, VT plen) {
    VT ret = 0;
    auto end = start + length;
    for (auto i = start; i < end; ++i)
        ret += in[i];
    return ret * plen;
}

template <typename VT> static inline VT getVar(const VT *in, uint32_t start, uint32_t length, VT plen, VT mean) {
    VT ret = 0;
    auto end = start + length;
    for (auto i = start; i < end; ++i)
        ret += (in[i] - mean) * (in[i] - mean);
    return ret * plen;
}

template <typename VTRes, typename VTArg> struct BatchNorm2DTestForward<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *in, const DenseMatrix<VTArg> *gamma,
                      const DenseMatrix<VTArg> *beta, const DenseMatrix<VTArg> *emaMean,
                      const DenseMatrix<VTArg> *emaVar, const VTArg eps, DCTX(dctx)) {

        auto start = 0;
        auto stop = in->getNumRows();
        auto size = in->getNumCols();

        VTArg x_hat = 0;
        auto off = 0;

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(stop, size, true);
        }

        for (uint32_t i = start; i < stop; i++) {
            for (uint32_t j = 0; j < size; j++) {
                off = i * size + j;
                x_hat = (in->getValues()[off] - emaMean->getValues()[i]) / std::sqrt(emaVar->getValues()[i] + eps);
                res->getValues()[off] = gamma->getValues()[i] * x_hat + beta->getValues()[i];
            }
        }
    }
};