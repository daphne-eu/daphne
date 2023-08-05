/*
 * Copyright 2023 The DAPHNE Consortium
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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNCALC_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNCALC_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/calc.hpp>
#include <runtime/local/kernels/BinaryOpCode.h>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTDataLhs, class DTDataRhs, typename VE>
struct ColumnCalc {
    static void apply(BinaryOpCode opCode, DTRes *& res, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTDataLhs, class DTDataRhs>
void columnBinary(BinaryOpCode opCode, DTRes *& res, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) {
    switch (ctx->getUserConfig().vector_extension) {
        case VectorExtensions::AVX512:
            ColumnCalc<DTRes, DTDataLhs, DTDataRhs, tsl::avx512>::apply(opCode, res, data_lhs, data_rhs, ctx);
            break;
        case VectorExtensions::AVX2:
            ColumnCalc<DTRes, DTDataLhs, DTDataRhs, tsl::avx2>::apply(opCode, res, data_lhs, data_rhs, ctx);
            break;
        //case VectorExtensions::SSE:
        //    ColumnCalc<DTRes, DTDataLhs, DTDataRhs, tsl::sse>::apply(opCode, res, data_lhs, data_rhs, ctx);
        //    break;
        case VectorExtensions::SCALAR:
            ColumnCalc<DTRes, DTDataLhs, DTDataRhs, tsl::scalar>::apply(opCode, res, data_lhs, data_rhs, ctx);
            break;
        default:
            throw std::runtime_error("Unknown vector extension");
    }
    //ColumnCalc<DTRes, DTDataLhs, DTDataRhs>::apply(opCode, res, data_lhs, data_rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT, typename VE>
struct ColumnCalc<tuddbs::Column<VT>, tuddbs::Column<VT>, tuddbs::Column<VT>, VE> {
    static void apply(BinaryOpCode opCode, tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data_lhs, const tuddbs::Column<VT> * data_rhs, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, VE>;
        switch (opCode) {
            case BinaryOpCode::ADD: {
                tuddbs::daphne_calc<ps, tsl::functors::add> calc;
                res = calc(data_lhs, data_rhs);   
                break;
            } 
            case BinaryOpCode::MUL: {
                tuddbs::daphne_calc<ps, tsl::functors::mul> calc;
                res = calc(data_lhs, data_rhs);
                break;
            } 
            case BinaryOpCode::AND: {
                tuddbs::daphne_calc<ps, tsl::functors::binary_and> calc;
                res = calc(data_lhs, data_rhs);
                break;
            } 
            case BinaryOpCode::OR: {
                tuddbs::daphne_calc<ps, tsl::functors::binary_or> calc;
                res = calc(data_lhs, data_rhs);
                break;
            } default: {
                throw std::runtime_error("Unsupported binary operator");
            }
        }
        
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNCALC_H