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
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <stdexcept>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResData, class DTLhsData, class DTRhsPos> struct ColProject {
    static void apply(DTResData *&resData, const DTLhsData *lhsData, const DTRhsPos *rhsPos, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResData, class DTLhsData, class DTRhsPos>
void colProject(DTResData *&resData, const DTLhsData *lhsData, const DTRhsPos *rhsPos, DCTX(ctx)) {
    ColProject<DTResData, DTLhsData, DTRhsPos>::apply(resData, lhsData, rhsPos, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos> struct ColProject<Column<VTData>, Column<VTData>, Column<VTPos>> {
    static void apply(Column<VTData> *&resData, const Column<VTData> *lhsData, const Column<VTPos> *rhsPos, DCTX(ctx)) {
        const size_t numLhsData = lhsData->getNumRows();
        const size_t numRhsPos = rhsPos->getNumRows();
        const size_t numResData = numRhsPos;

        if (resData == nullptr)
            resData = DataObjectFactory::create<Column<VTData>>(numResData, false);

        const VTData *valuesLhsData = lhsData->getValues();
        const VTPos *valuesRhsPos = rhsPos->getValues();
        VTData *valuesResData = resData->getValues();
        for (size_t r = 0; r < numRhsPos; r++) {
            const VTPos pos = valuesRhsPos[r];
            if(pos < 0 || pos > numLhsData)
                // TODO more details
                throw std::runtime_error("colProject: out-of-bounds access");
            valuesResData[r] = valuesLhsData[pos];
        }
    }
};