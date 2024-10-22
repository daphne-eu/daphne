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

template<class DTRes, class DTArg>
struct BatchNorm2DBackward {
    static void apply(  DTRes *&dX, DTRes *&dGamma, DTRes *&dBeta,
                        const DTArg *mean, const DTArg *invVar, 
                        const DTArg *in, const DTArg *dout,
                        const DTArg *gamma, const typename DTArg::VT eps, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void batchNorm2DBackward(   DTRes *&dX, DTRes *&dGamma, DTRes *&dBeta,
                            const DTArg *mean, const DTArg *invVar, 
                            const DTArg *in, const DTArg *dout, 
                            const DTArg *gamma, const typename DTArg::VT eps, DCTX(dctx)) {
    BatchNorm2DBackward<DTRes, DTArg>::apply(dX, dGamma, dBeta, mean, invVar, in, dout, gamma, eps, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg>
struct BatchNorm2DBackward<DenseMatrix<VTRes>, DenseMatrix<VTArg>>
{
    static void 
    apply(  DenseMatrix<VTRes> *&dX,
            DenseMatrix<VTRes> *&dGamma,
            DenseMatrix<VTRes> *&dBeta,
            const DenseMatrix<VTArg> *mean,
            const DenseMatrix<VTArg> *invVar,
            const DenseMatrix<VTArg> *in,
            const DenseMatrix<VTArg> *dout, 
            const DenseMatrix<VTArg> *gamma,
            const VTArg eps, DCTX(dctx))
    {
        
        auto start = 0;
        auto stop = in->getNumRows();
        auto CHW = in->getNumCols();
        auto C = gamma->getNumRows();
        auto HW = CHW / C;
        VTArg m = stop * HW;

        auto half = static_cast<typename DenseMatrix<VTArg>::VT>(1) / static_cast<typename DenseMatrix<VTArg>::VT>(2);
        auto const_2_m = static_cast<typename DenseMatrix<VTArg>::VT>(2) / m;
        // auto const_1_m = static_cast<typename DenseMatrix<VTArg>::VT>(1) / m;

        auto off = 0;
        VTArg sum_dBeta = 0, sum_dGamma = 0, dVar = 0, dMean = 0, dX_hat = 0;
        /* double sum_dBeta = 0., sum_dGamma = 0.;
        double dVar = 0.; 
        double dMean = 0.; 
        double dX_hat = 0.; */

        if (dX == nullptr)
            dX = DataObjectFactory::create<DenseMatrix<VTArg>>(stop, CHW, true);
        if (dGamma == nullptr)
            dGamma = DataObjectFactory::create<DenseMatrix<VTArg>>(C, 1, true);
        if (dBeta == nullptr)
            dBeta = DataObjectFactory::create<DenseMatrix<VTArg>>(C, 1, true);
        
        for(uint32_t c = 0; c < C; c++)
        {
            sum_dBeta = 0, sum_dGamma = 0, dVar = 0, dMean = 0, dX_hat = 0;
            for(uint32_t i = start; i < stop; i++)
                for(uint32_t j = 0; j < HW; j++)
                {
                    off = i * CHW + c * HW + j;
                    sum_dBeta += dout->getValues()[off];
                    sum_dGamma += dout->getValues()[off] * (in->getValues()[off] - mean->getValues()[c]) * invVar->getValues()[c];
                    dX_hat = dout->getValues()[off] * gamma->getValues()[c];
                    dVar -= dX_hat
                            * (in->getValues()[off] - mean->getValues()[c])
                            * half * std::pow(invVar->getValues()[c], 3);
                }
            dBeta->getValues()[c] = sum_dBeta;
            dGamma->getValues()[c] = sum_dGamma;
          
            for(uint32_t i = start; i < stop; i++)
                for(uint32_t j = 0; j < HW; j++)
                {
                    off = i * CHW + c * HW + j;
                    dX_hat = dout->getValues()[off] * gamma->getValues()[c];
                    dMean += dX_hat * (-invVar->getValues()[c])
                            + dVar * (-const_2_m) * (in->getValues()[off] - mean->getValues()[c]);            
                }
            for(uint32_t i = start; i < stop; i++)
                for(uint32_t j = 0; j < HW; j++)
                {
                    off = i * CHW + c * HW + j;
                    dX_hat = dout->getValues()[off] * gamma->getValues()[c];
                    dX->getValues()[off] = dX_hat * invVar->getValues()[c]
                                    + dVar * const_2_m * (in->getValues()[off] - mean->getValues()[c])
                                    + dMean / m;
                }  
        }
    }
};