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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/CastObj.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResData, class DTLhsData, class DTRhsData> struct ColCalcBinary {
    static void apply(BinaryOpCode opCode, DTResData *&resData, const DTLhsData *lhsData, const DTRhsData *rhsData,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResData, class DTLhsData, class DTRhsData>
void colCalcBinary(BinaryOpCode opCode, DTResData *&resData, const DTLhsData *lhsData, const DTRhsData *rhsData,
                   DCTX(ctx)) {
    ColCalcBinary<DTResData, DTLhsData, DTRhsData>::apply(opCode, resData, lhsData, rhsData, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTResData, typename VTLhsData, typename VTRhsData>
struct ColCalcBinary<Column<VTResData>, Column<VTLhsData>, Column<VTRhsData>> {
    static void apply(BinaryOpCode opCode, Column<VTResData> *&resData, const Column<VTLhsData> *lhsData,
                      const Column<VTRhsData> *rhsData, DCTX(ctx)) {
        DenseMatrix<VTResData> * resDataMat = nullptr;
        DenseMatrix<VTLhsData> * lhsDataMat = nullptr;
        DenseMatrix<VTRhsData> * rhsDataMat = nullptr;
        castObj<DenseMatrix<VTLhsData>>(lhsDataMat, lhsData, ctx);
        castObj<DenseMatrix<VTRhsData>>(rhsDataMat, rhsData, ctx);
        ewBinaryMat(opCode, resDataMat, lhsDataMat, rhsDataMat, ctx);
        castObj<Column<VTResData>>(resData, resDataMat, ctx);
    }
};