/*
 * Copyright 2021 The DAPHNE Consortium
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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<typename VTRes, class DTArg>
struct CastObjSca {
    static VTRes apply(const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<typename VTRes, class DTArg>
VTRes castObjSca(const DTArg * arg, DCTX(ctx)) {
    return CastObjSca<VTRes, DTArg>::apply(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Scalar <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTArg>
struct CastObjSca<VTRes, DenseMatrix<VTArg>> {
    static VTRes apply(const DenseMatrix<VTArg> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if(numCols != 1 || numRows != 1)
            throw std::runtime_error("Cast matrix to scalar: matrix shape should be 1x1");
        return static_cast<VTRes>(*arg->getValues());
    }
};

// ----------------------------------------------------------------------------
//  Scalar <- Frame
// ----------------------------------------------------------------------------

template<typename VTRes>
struct CastObjSca<VTRes, Frame> {
    static VTRes apply(const Frame * arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        if(numCols != 1 || numRows != 1)
            throw std::runtime_error("Cast frame to scalar: frame shape should be 1x1");
        
        VTRes res = VTRes(0);
        auto colType = static_cast<unsigned int>(arg->getColumnType(0));
        const void * resVal = arg->getColumnRaw(0);
        // Cast void* to the largest column type width (split integer and floating point interpretations) 
        // and then final cast to VTRes. This way we avoid DenseMatrix creation in Frame::getColumn().
        // TODO It is dangerous to treat the value type code as an integer here,
        // since this can easily break if we change the value type codes.
        if(colType <= 5U)
            res = static_cast<VTRes>(*reinterpret_cast<const int64_t*>(resVal));
        else if(colType <= 7U)
            res = static_cast<VTRes>(*reinterpret_cast<const double*>(resVal));
        else
            throw std::runtime_error("CastObjSca::apply: unknown value type code");

        return res;
    }
};
