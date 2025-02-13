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
 #include <unordered_set>
 
 #include <cstddef>
 
 // ****************************************************************************
 // Struct for partial template specialization
 // ****************************************************************************
 
 template <class DTResLhsPos, class DTLhsData, class DTRhsData> struct ColSemiJoin {
     static void apply(DTResLhsPos *&resLhsPos, const DTLhsData *lhsData, const DTRhsData *rhsData,
                       int64_t numRes, DCTX(ctx)) = delete;
 };
 
 // ****************************************************************************
 // Convenience function
 // ****************************************************************************
 
 template <class DTResLhsPos, class DTLhsData, class DTRhsData>
 void colSemiJoin(DTResLhsPos *&resLhsPos, const DTLhsData *lhsData, const DTRhsData *rhsData,
              int64_t numRes, DCTX(ctx)) {
     ColSemiJoin<DTResLhsPos, DTLhsData, DTRhsData>::apply(resLhsPos, lhsData, rhsData, numRes, ctx);
 }
 
 // ****************************************************************************
 // (Partial) template specializations for different data/value types
 // ****************************************************************************
 
 // ----------------------------------------------------------------------------
 // Column <- Column, Column
 // ----------------------------------------------------------------------------
 
 template <typename VTData, typename VTPos>
 struct ColSemiJoin<Column<VTPos>, Column<VTData>, Column<VTData>> {
     static void apply(Column<VTPos> *&resLhsPos, const Column<VTData> *lhsData,
                       const Column<VTData> *rhsData, int64_t numRes, DCTX(ctx)) {
         const size_t numLhsData = lhsData->getNumRows();
         const size_t numRhsData = rhsData->getNumRows();
 
         if (numRes == -1)
             // TODO maybe assume PK-FK join
             numRes = numLhsData * numRhsData;
 
         if (resLhsPos == nullptr)
             resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);
         VTPos *valuesResLhsPos = resLhsPos->getValues();
 
         // Build phase.
         std::unordered_set<VTData> ht;
         const VTData *valuesRhsData = rhsData->getValues();
         for(size_t r = 0; r < numRhsData; r++)
             ht.insert(valuesRhsData[r]);
 
         // Probe phase.
         const VTData *valuesLhsData = lhsData->getValues();
         size_t posRes = 0;
         for (size_t r = 0; r < numLhsData; r++)
             if(ht.count(valuesLhsData[r]))
                 valuesResLhsPos[posRes++] = r;
 
         resLhsPos->shrinkNumRows(posRes);
     }
 };