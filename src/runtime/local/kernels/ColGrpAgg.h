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

template <class DTRes, class DTData, class DTGrpIds> struct ColGrpAgg {
    static void apply(AggOpCode opCode, DTRes *&res, const DTData *data, const DTGrpIds *grpIds, size_t numDistinct, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTData, class DTGrpIds>
void colGrpAgg(AggOpCode opCode, DTRes *&res, const DTData *data, const DTGrpIds *grpIds, size_t numDistinct, DCTX(ctx)) {
    ColGrpAgg<DTRes, DTData, DTGrpIds>::apply(opCode, res, data, grpIds, numDistinct, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos> struct ColGrpAgg<Column<VTData>, Column<VTData>, Column<VTPos>> {
    static void apply(AggOpCode opCode, Column<VTData> *&res, const Column<VTData> *data, const Column<VTPos> *grpIds,
                      size_t numDistinct, DCTX(ctx)) {
        const size_t numData = data->getNumRows();

        if (numData != grpIds->getNumRows())
            throw std::runtime_error("input data and input group ids must have the same number of elements");

        const VTData *valuesData = data->getValues();
        const VTPos *valuesGrpIds = grpIds->getValues();

        if (res == nullptr)
            res = DataObjectFactory::create<Column<VTData>>(numDistinct, true);

        VTData *valuesRes = res->getValues();

        EwBinaryScaFuncPtr<VTData, VTData, VTData> func;
        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
            func = getEwBinaryScaFuncPtr<VTData, VTData, VTData>(AggOpCodeUtils::getBinaryOpCode(opCode));

            for (size_t r = 0; r < numData; r++) {
                VTPos grpId = valuesGrpIds[r];
                if (grpId < 0 || grpId > numDistinct)
                    throw std::runtime_error("out-of-bounds access");
                valuesRes[grpId] = func(valuesRes[grpId], valuesData[r], ctx);
            }
        } else
            throw std::runtime_error("unsupported op code");

        // std::cerr << "colGrpAgg: res: " << std::endl;
        // res->print(std::cerr);
    }
};