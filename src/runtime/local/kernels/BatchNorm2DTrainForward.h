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
struct BatchNorm2DTrainForward {
    static void apply(  DTRes *&res, 
                        DTRes *&new_emaMean, DTRes *&new_emaVar,
                        DTRes *&mean, DTRes *&invVar, 
                        const DTArg *in, 
                        const DTArg *gamma, const DTArg *beta,
                        const DTArg *emaMean, const DTArg *emaVar, 
                        const typename DTArg::VT eps, const typename DTArg::VT mu,
                        DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void batchNorm2DTrainForward(   DTRes *&res, 
                            DTRes *&new_emaMean, DTRes *&new_emaVar,
                            DTRes *&mean, DTRes *&invVar, 
                            const DTArg *in, 
                            const DTArg *gamma, const DTArg *beta,
                            const DTArg *emaMean, const DTArg *emaVar, 
                            const typename DTArg::VT eps, const typename DTArg::VT mu,
                            DCTX(dctx)) {
    BatchNorm2DTrainForward<DTRes, DTArg>::apply(res, new_emaMean, new_emaVar, mean, invVar, in, gamma, beta, emaMean, emaVar, eps, mu, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg>
struct BatchNorm2DTrainForward<DenseMatrix<VTRes>, DenseMatrix<VTArg>>
{
    static void 
    apply(  DenseMatrix<VTRes> *&res,
            DenseMatrix<VTRes> *&new_emaMean,
            DenseMatrix<VTRes> *&new_emaVar,
            DenseMatrix<VTRes> *&Mean,
            DenseMatrix<VTRes> *&invVar,
            const DenseMatrix<VTArg> *in,
            const DenseMatrix<VTArg> *gamma, 
            const DenseMatrix<VTArg> *beta,
            const DenseMatrix<VTArg> *emaMean,
            const DenseMatrix<VTArg> *emaVar,
            const VTArg eps, const VTArg mu, DCTX(dctx))
    {
        
        auto start = 0;
        auto stop = in->getNumRows();
        auto CHW = in->getNumCols();
        auto C = gamma->getNumRows();
        auto HW = CHW / C;

        VTArg x_hat = 0;
        VTArg mean = 0;
        VTArg var = 0;
        auto off = 0;

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(stop, CHW, true);
        if (new_emaMean == nullptr)
            new_emaMean = DataObjectFactory::create<DenseMatrix<VTArg>>(C, C, true);
        if (new_emaVar == nullptr)
            new_emaVar = DataObjectFactory::create<DenseMatrix<VTArg>>(C, C, true);
        if (Mean == nullptr)
            Mean = DataObjectFactory::create<DenseMatrix<VTArg>>(C, C, true);
        if (invVar == nullptr)
            invVar = DataObjectFactory::create<DenseMatrix<VTArg>>(C, C, true);
        
        for(uint32_t c = 0; c < C; c++)
        {
            mean = 0;
            for(uint32_t i = start; i < stop; i++)
                for(uint32_t j = 0; j < HW; j++)
                {
                    off = i * CHW + c * HW + j;
                    mean =  mean + in->getValues()[off] / (stop * HW);
                }
            // std::cout<<mean<<std::endl;
            var = 0;
            for(uint32_t i = start; i < stop; i++)
                for(uint32_t j = 0; j < HW; j++)
                {
                    off = i * CHW + c * HW + j;
                    var =  var + std::pow((in->getValues()[off] - mean), 2) / (stop * HW);
                }
            // std::cout<<var<<std::endl;
            Mean->getValues()[c] = mean;
            invVar->getValues()[c] = 1 / std::sqrt(var + eps);
            new_emaMean->getValues()[c] = (1 - mu) * emaMean->getValues()[c] + mu * mean;
            new_emaVar->getValues()[c] = (1 - mu) * emaVar->getValues()[c] + mu * var;
            for(uint32_t i = start; i < stop; i++)
            {
                for(uint32_t j = 0; j < HW; j++)
                {
                    off = i * CHW + c * HW + j;
                    x_hat = (in->getValues()[off] - mean) / std::sqrt(var + eps);
                    res->getValues()[off] = gamma->getValues()[c] * x_hat + beta->getValues()[c];
                }
            }

        }
    }
};