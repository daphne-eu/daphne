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

#include "CmpOpCode.h"
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/CmpOpCode.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResPos, class DTLhsData, typename VTRhsData> struct ColSelectCmp {
    static void apply(CmpOpCode opCode, DTResPos *&resPos, const DTLhsData *lhsData, VTRhsData rhsData, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResPos, class DTLhsData, typename VTRhsData>
void colSelectCmp(CmpOpCode opCode, DTResPos *&resPos, const DTLhsData *lhsData, VTRhsData rhsData, DCTX(ctx)) {
    ColSelectCmp<DTResPos, DTLhsData, VTRhsData>::apply(opCode, resPos, lhsData, rhsData, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, scalar
// ----------------------------------------------------------------------------

template <typename VTPos, typename VTLhsData, typename VTRhsData> struct ColSelectCmp<Column<VTPos>, Column<VTLhsData>, VTRhsData> {
    static void apply(CmpOpCode opCode, Column<VTPos> *&resPos, const Column<VTLhsData> *lhsData, VTRhsData rhsData, DCTX(ctx)) {
        const size_t numLhsData = lhsData->getNumRows();

        if (resPos == nullptr)
            resPos = DataObjectFactory::create<Column<VTPos>>(numLhsData, false);

        const VTLhsData *valuesLhsData = lhsData->getValues();
        VTPos *valuesResPos = resPos->getValues();

        bool (*func)(VTLhsData, VTRhsData) = nullptr;
        switch (opCode) {
        case CmpOpCode::EQ:
            func = [](VTLhsData lhs, VTRhsData rhs) { return lhs == rhs; };
            break;
        case CmpOpCode::NEQ:
            func = [](VTLhsData lhs, VTRhsData rhs) { return lhs != rhs; };
            break;
        case CmpOpCode::GT:
            func = [](VTLhsData lhs, VTRhsData rhs) { return lhs > rhs; };
            break;
        case CmpOpCode::GE:
            func = [](VTLhsData lhs, VTRhsData rhs) { return lhs >= rhs; };
            break;
        case CmpOpCode::LT:
            func = [](VTLhsData lhs, VTRhsData rhs) { return lhs < rhs; };
            break;
        case CmpOpCode::LE:
            func = [](VTLhsData lhs, VTRhsData rhs) { return lhs <= rhs; };
            break;
        }

        size_t numResPos = 0;
        for (size_t r = 0; r < numLhsData; r++)
            if (func(valuesLhsData[r], rhsData)) {
                valuesResPos[numResPos] = r;
                numResPos++;
            }

        resPos->shrinkNumRows(numResPos);
    }
};