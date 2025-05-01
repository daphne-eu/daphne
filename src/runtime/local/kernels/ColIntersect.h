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

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResPos, class DTLhsPos, class DTRhsPos> struct ColIntersect {
    static void apply(DTResPos *&resPos, const DTLhsPos *lhsPos, const DTRhsPos *rhsPos, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResPos, class DTLhsPos, class DTRhsPos>
void colIntersect(DTResPos *&resPos, const DTLhsPos *lhsPos, const DTRhsPos *rhsPos, DCTX(ctx)) {
    ColIntersect<DTResPos, DTLhsPos, DTRhsPos>::apply(resPos, lhsPos, rhsPos, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTPos> struct ColIntersect<Column<VTPos>, Column<VTPos>, Column<VTPos>> {
    static void apply(Column<VTPos> *&resPos, const Column<VTPos> *lhsPos, const Column<VTPos> *rhsPos, DCTX(ctx)) {
        const size_t numLhsPos = lhsPos->getNumRows();
        const size_t numRhsPos = rhsPos->getNumRows();

        if (resPos == nullptr)
            resPos = DataObjectFactory::create<Column<VTPos>>(std::min(numLhsPos, numRhsPos), false);

        const VTPos *valuesLhsPos = lhsPos->getValues();
        const VTPos *valuesRhsPos = rhsPos->getValues();
        const VTPos *valuesLhsPosEnd = valuesLhsPos + numLhsPos;
        const VTPos *valuesRhsPosEnd = valuesRhsPos + numRhsPos;
        VTPos *valuesResPos = resPos->getValues();
        VTPos *valuesResPosBeg = valuesResPos;

        while (valuesLhsPos < valuesLhsPosEnd && valuesRhsPos < valuesRhsPosEnd) {
            if (*valuesLhsPos < *valuesRhsPos)
                valuesLhsPos++;
            else if (*valuesRhsPos < *valuesLhsPos)
                valuesRhsPos++;
            else { // *valuesLhsPos == *valuesRhsPos
                *valuesResPos = *valuesLhsPos;
                valuesLhsPos++;
                valuesRhsPos++;
                valuesResPos++;
            }
        }

        resPos->shrinkNumRows(valuesResPos - valuesResPosBeg);
    }
};