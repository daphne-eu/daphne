/*
 * Copyright 2025 The DAPHNE Consortium
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
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/CastObj.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTData, class DTGrpIds> struct ColAggGrp {
    static void apply(AggOpCode opCode, DTRes *&res, const DTData *data, const DTGrpIds *grpIds, size_t numDistinct,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTData, class DTGrpIds>
void colAggGrp(AggOpCode opCode, DTRes *&res, const DTData *data, const DTGrpIds *grpIds, size_t numDistinct,
               DCTX(ctx)) {
    ColAggGrp<DTRes, DTData, DTGrpIds>::apply(opCode, res, data, grpIds, numDistinct, ctx);
}

// ****************************************************************************
// Utility function
// ****************************************************************************

template <typename VTData, typename VTPos>
void applyAgg(AggOpCode opCode, const VTPos *valuesGrpIds, const VTData *valuesData, VTData *valuesRes,
              const size_t numData, size_t numDistinct, DCTX(ctx)) {
    EwBinaryScaFuncPtr<VTData, VTData, VTData> func;
    func = getEwBinaryScaFuncPtr<VTData, VTData, VTData>(AggOpCodeUtils::getBinaryOpCode(opCode));

    for (size_t r = 0; r < numData; r++) {
        VTPos grpId = valuesGrpIds[r];
        if (grpId < 0 || grpId > numDistinct)
            throw std::runtime_error("out-of-bounds access");
        valuesRes[grpId] = func(valuesRes[grpId], valuesData[r], ctx);
    }
}

template <typename VTData, typename VTPos>
void applyAggOptimisticSplit(AggOpCode opCode, const VTPos *valuesGrpIds, const VTData *valuesData, VTData *valuesRes,
              const size_t numData, size_t numDistinct, DCTX(ctx)) {

    // split result value 
    using HalfTypeT = typename ValueTypeUtils::HalfType<VTData>::type;
    auto resCommon = DataObjectFactory::create<Column<HalfTypeT>>(numDistinct, false);
    HalfTypeT *valuesResCom =  resCommon->getValues();
    std::fill(valuesResCom, valuesResCom + numDistinct, AggOpCodeUtils::getNeutral<HalfTypeT>(opCode));
    
    auto funcOpt = getEwBinaryScaFuncPtr<HalfTypeT, VTData, HalfTypeT>(AggOpCodeUtils::optimisticSplitCommon(opCode));
    auto funcExp = getEwBinaryScaFuncPtr<VTData, VTData, HalfTypeT>(AggOpCodeUtils::optimisticSplitExcept(opCode));
    auto funcOverflow = getEwBinaryScaFuncPtr<bool, VTData, HalfTypeT>(AggOpCodeUtils::optimisticSplitOverflow(opCode));

    // store the operand result. In case of overflow the current valuesResCom[grpId] will be safe.
    HalfTypeT tmp = 0; 

    for (size_t r = 0; r < numData; r++) {
        VTPos grpId = valuesGrpIds[r];
        if (grpId < 0 || grpId > numDistinct)
            throw std::runtime_error("out-of-bounds access");

        tmp = funcOpt(valuesData[r], valuesResCom[grpId], ctx);
        bool overflow = funcOverflow(valuesData[r], tmp, ctx);
        if(overflow){
            // if overflow update the result directly.
            valuesRes[grpId] += funcExp(valuesData[r],  valuesResCom[grpId], ctx);
        }
        // If overflow, valuesResCom[grpId] is added to result. So set it to 0 otherwise to tmp.
        // Note: this default value (HalfTypeT(0)) may need to be updated based on opCode.
        valuesResCom[grpId] = overflow ? HalfTypeT(0) : tmp;
    }

    for(size_t r = 0; r < numDistinct; r++){
        valuesRes[r] += valuesResCom[r] > 0 ? static_cast<VTData>(valuesResCom[r]) : 0;
    }
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos> struct ColAggGrp<Column<VTData>, Column<VTData>, Column<VTPos>> {
    static void apply(AggOpCode opCode, Column<VTData> *&res, const Column<VTData> *data, const Column<VTPos> *grpIds,
                      size_t numDistinct, DCTX(ctx)) {
        const size_t numData = data->getNumRows();

        if (numData != grpIds->getNumRows())
            throw std::runtime_error("input data and input group ids must have the same number of elements");

        const VTData *valuesData = data->getValues();
        const VTPos *valuesGrpIds = grpIds->getValues();

        if (res == nullptr)
            res = DataObjectFactory::create<Column<VTData>>(numDistinct, false);

        VTData *valuesRes = res->getValues();

        // Initialize the accumulator of each group with the neutral element of the aggregation function.
        std::fill(valuesRes, valuesRes + numDistinct, AggOpCodeUtils::getNeutral<VTData>(opCode));

        // Perform the grouped aggregation.
        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
            if(opCode == AggOpCode::SUM){
                if constexpr(isSupportOptimistic<AggOpCode::SUM, VTData, VTData>){
                    applyAggOptimisticSplit<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData, numDistinct, ctx);
                }
                else
                {
                    applyAgg<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData, numDistinct, ctx);
                }
            }
            else{
                applyAgg<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData, numDistinct, ctx);
            }
        } else
            throw std::runtime_error("unsupported op code");
    }
};