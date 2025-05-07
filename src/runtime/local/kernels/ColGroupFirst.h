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

#include <unordered_map>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResGrpIds, class DTResReprPos, class DTArgData> struct ColGroupFirst {
    static void apply(DTResGrpIds *&resGrpIds, DTResReprPos *&resReprPos, const DTArgData *argData, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResGrpIds, class DTResReprPos, class DTArgData>
void colGroupFirst(DTResGrpIds *&resGrpIds, DTResReprPos *&resReprPos, const DTArgData *argData, DCTX(ctx)) {
    ColGroupFirst<DTResGrpIds, DTResReprPos, DTArgData>::apply(resGrpIds, resReprPos, argData, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column, Column <- Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos> struct ColGroupFirst<Column<VTPos>, Column<VTPos>, Column<VTData>> {
    static void apply(Column<VTPos> *&resGrpIds, Column<VTPos> *&resReprPos, const Column<VTData> *argData, DCTX(ctx)) {
        const size_t numArgData = argData->getNumRows();

        if (resGrpIds == nullptr)
            resGrpIds = DataObjectFactory::create<Column<VTPos>>(numArgData, false);
        if (resReprPos == nullptr)
            resReprPos = DataObjectFactory::create<Column<VTPos>>(numArgData, false);
        VTPos *valuesResGrpIds = resGrpIds->getValues();
        VTPos *valuesResReprPos = resReprPos->getValues();
        VTPos *valuesResReprPosBeg = valuesResReprPos;

        const VTData *valuesArgData = argData->getValues();
        std::unordered_map<VTData, VTPos> grpIds;
        for (size_t r = 0; r < numArgData; r++) {
            VTPos &grpId = grpIds[valuesArgData[r]];
            if (!grpId) { // the value was not found
                grpId = grpIds.size();
                *valuesResReprPos = r;
                valuesResReprPos++;
            }
            *valuesResGrpIds = grpId - 1; // -1 because we use a zero entry in ht to indicate a newly created entry
            valuesResGrpIds++;
        }

        resReprPos->shrinkNumRows(valuesResReprPos - valuesResReprPosBeg);
    }
};