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
                      const bool optimisticSplit, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTData, class DTGrpIds>
void colAggGrp(AggOpCode opCode, DTRes *&res, const DTData *data, const DTGrpIds *grpIds, size_t numDistinct,
               const bool optimisticSplit, DCTX(ctx)) {
    ColAggGrp<DTRes, DTData, DTGrpIds>::apply(opCode, res, data, grpIds, numDistinct, optimisticSplit, ctx);
}

// ****************************************************************************
// Utility functions
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

// ****************************************************************************
// Utility functions for optimistic splitting
// ****************************************************************************

// Template to specifying HalfType value types.
template <typename ValueType> struct HalfType {};

#define HalfTypeDefine(VT, VTHalf)                                                                                     \
    template <> struct HalfType<VT> {                                                                                  \
        using type = VTHalf;                                                                                           \
    };

HalfTypeDefine(int64_t, int32_t) HalfTypeDefine(uint64_t, uint32_t) HalfTypeDefine(int32_t, int16_t)
    HalfTypeDefine(uint32_t, uint16_t) HalfTypeDefine(double, float)

#undef HalfTypeDefine

    // Main function for grouped aggregation with optimistic splitting
    template <typename VTData, typename VTPos>
    void applyAggOptimisticSplit(AggOpCode opCode, const VTPos *valuesGrpIds, const VTData *valuesData,
                                 VTData *valuesRes, const size_t numData, size_t numDistinct, DCTX(ctx)) {

    using HalfTypeT = typename HalfType<VTData>::type;

    // Generate common part on the result.
    auto resCommon = DataObjectFactory::create<Column<HalfTypeT>>(numDistinct, false);
    HalfTypeT *valuesResCom = resCommon->getValues();
    std::fill(valuesResCom, valuesResCom + numDistinct, AggOpCodeUtils::getNeutral<HalfTypeT>(opCode));

    // This value temporary stores the operand result for common part.
    // In case of overflow the current valuesResCom[grpId] will be safe.
    HalfTypeT tmp = 0;

    for (size_t r = 0; r < numData; r++) {
        VTPos grpId = valuesGrpIds[r];
        if (grpId < 0 || grpId > numDistinct)
            throw std::runtime_error("out-of-bounds access");

        // Apply the agg function, and compute the result for common part.
        tmp = static_cast<HalfTypeT>(valuesData[r]) + valuesResCom[grpId];

        // Check the agg result for overflow.
        // This condition should support both positive and negative values.
        bool overflow = !((valuesData[r] >= 0) ^ (tmp < valuesData[r]));

        // If overflow, update the result directly.
        if (overflow) {
            valuesRes[grpId] += valuesData[r] + static_cast<VTData>(valuesResCom[grpId]);
        }

        // If overflow, then valuesResCom[grpId] was added to the final result.
        // Otherwise its new value is tmp.
        valuesResCom[grpId] = overflow ? HalfTypeT(0) : tmp;
    }

    for (size_t r = 0; r < numDistinct; r++) {
        valuesRes[r] += valuesResCom[r] != 0 ? static_cast<VTData>(valuesResCom[r]) : 0;
    }
}

/**
 * Template constant specifying if the AggOpCode supports arguments of the given
 * value types with optimistic split.
 * Used in the ColAggGrp function to check if optimistic split is supported for agg and arguments.
 */
template <AggOpCode op, typename VData> static constexpr bool supportOptimistic = false;

#define SUPPORT(Op, VT) template <> constexpr bool supportOptimistic<AggOpCode::Op, VT> = true;

SUPPORT(SUM, int64_t)
SUPPORT(SUM, uint64_t)
SUPPORT(SUM, int32_t)
SUPPORT(SUM, uint32_t)

#undef SUPPORT
// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos> struct ColAggGrp<Column<VTData>, Column<VTData>, Column<VTPos>> {
    static void apply(AggOpCode opCode, Column<VTData> *&res, const Column<VTData> *data, const Column<VTPos> *grpIds,
                      size_t numDistinct, const bool optimisticSplit, DCTX(ctx)) {
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
            if ((opCode == AggOpCode::SUM) & optimisticSplit) {
                if constexpr (supportOptimistic<AggOpCode::SUM, VTData>) {
                    applyAggOptimisticSplit<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData,
                                                           numDistinct, ctx);
                } else {
                    applyAgg<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData, numDistinct, ctx);
                }
            } else {
                applyAgg<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData, numDistinct, ctx);
            }
        } else
            throw std::runtime_error("unsupported op code");
    }
};