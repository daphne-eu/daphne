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
                      bool allowOptimisticSplitting, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTData, class DTGrpIds>
void colAggGrp(AggOpCode opCode, DTRes *&res, const DTData *data, const DTGrpIds *grpIds, size_t numDistinct,
               bool allowOptimisticSplitting, DCTX(ctx)) {
    ColAggGrp<DTRes, DTData, DTGrpIds>::apply(opCode, res, data, grpIds, numDistinct, allowOptimisticSplitting, ctx);
}

// ****************************************************************************
// Utility functions for the default implementation
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
// Utility functions for the implementation using optimistic splitting
// ****************************************************************************

// The implementation of this kernel using optimistic splitting is inspired by Section III of the following paper:
//
//   Tim Gubner, Viktor Leis, Peter A. Boncz: Efficient Query Processing with Optimistically Compressed Hash Tables &
//   Strings in the USSR. ICDE 2020: 301-312
//
// The basic idea is that most updates of the per-group accumulators will only change the lower half of the value. Thus,
// the kernel works primarily on an array of the lower halves of the accumulators (which has half the size of the final
// result and is, thus, more likely to fit into the processor cache). Only in the rare case of an overflow, the final
// accumulator of the full value type size is updated. Overall, this yields a more cache-efficient behavior.

/**
 * @brief Template providing a value type half the size of the given value type.
 */
template <typename ValueType> struct HalfType {};

#define DefineHalfType(VT, VTHalf)                                                                                     \
    template <> struct HalfType<VT> {                                                                                  \
        using type = VTHalf;                                                                                           \
    };
DefineHalfType(int64_t, int32_t);
DefineHalfType(uint64_t, uint32_t);
DefineHalfType(int32_t, int16_t);
DefineHalfType(uint32_t, uint16_t);
#undef DefineHalfType

// Utility function for grouped aggregation with optimistic splitting.
template <typename VTData, typename VTPos>
void applyAggOptimisticSplitting(AggOpCode opCode, const VTPos *valuesGrpIds, const VTData *valuesData,
                                 VTData *valuesRes, const size_t numData, size_t numDistinct, DCTX(ctx)) {
    using VTDataHalf = typename HalfType<VTData>::type;

    // Create an array of the lower halves of the accumulators.
    VTDataHalf *valuesResLo = new VTDataHalf[numDistinct];
    std::fill(valuesResLo, valuesResLo + numDistinct, AggOpCodeUtils::getNeutral<VTDataHalf>(opCode));

    // This value temporarily stores the operand result for the lower half.
    // In case of overflow the current valuesResLo[grpId] will be safe.
    VTDataHalf tmp = 0;

    for (size_t r = 0; r < numData; r++) {
        VTPos grpId = valuesGrpIds[r];
        if (grpId < 0 || grpId > numDistinct)
            throw std::runtime_error("out-of-bounds access");

        // Apply the agg function, and compute the result for the lower part.
        tmp = static_cast<VTDataHalf>(valuesData[r]) + valuesResLo[grpId];

        // Check the agg result for overflow.
        // This condition should support both positive and negative values.
        bool overflow = !((valuesData[r] >= 0) ^ (tmp < valuesData[r]));

        if (overflow) {
            // If there was an overflow, update the result directly and reset the lower part of the accumulator.
            valuesRes[grpId] += valuesData[r] + static_cast<VTData>(valuesResLo[grpId]);
            valuesResLo[grpId] = AggOpCodeUtils::getNeutral<VTDataHalf>(opCode);
        } else
            // Otherwise, set the lower part of the accumulator to tmp.
            valuesResLo[grpId] = tmp;
    }

    // Add the remaining lower parts of the accumulators to the final accumulators.
    for (size_t r = 0; r < numDistinct; r++)
        valuesRes[r] += static_cast<VTData>(valuesResLo[r]);

    delete[] valuesResLo;
}

/**
 * @brief Template constant specifying if the AggOpCode supports arguments of the given value type with optimistic
 * splitting.
 *
 * Used in the ColAggGrp function to check if optimistic splitting is supported for the aggregation function and
 * arguments.
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
                      size_t numDistinct, bool allowOptimisticSplitting, DCTX(ctx)) {
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
            if ((opCode == AggOpCode::SUM) && allowOptimisticSplitting) {
                if constexpr (supportOptimistic<AggOpCode::SUM, VTData>)
                    applyAggOptimisticSplitting<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData,
                                                               numDistinct, ctx);
                else
                    applyAgg<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData, numDistinct, ctx);
            } else
                applyAgg<VTData, VTPos>(opCode, valuesGrpIds, valuesData, valuesRes, numData, numDistinct, ctx);
        } else
            throw std::runtime_error("unsupported op code");
    }
};