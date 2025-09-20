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
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/CastObj.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct ColAggAll {
    static void apply(AggOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg> void colAggAll(AggOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    ColAggAll<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct ColAggAll<Column<VTRes>, Column<VTArg>> {
    static void apply(AggOpCode opCode, Column<VTRes> *&res, const Column<VTArg> *arg, DCTX(ctx)) {
        DenseMatrix<VTArg> *argMat = nullptr;
        castObj<DenseMatrix<VTArg>>(argMat, arg, ctx);
        VTRes resSca = aggAll<VTRes>(opCode, argMat, ctx);
        if (res == nullptr)
            res = DataObjectFactory::create<Column<VTRes>>(1, false);
        res->getValues()[0] = resSca;
    }
};