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

template<class DTRes, class DTDataLhs, class DTDataRhs>
struct ColumnCalc {
    static void apply(BinaryOpCode opCode, DTRes *& res, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTDataLhs, class DTDataRhs>
void columnBinary(BinaryOpCode opCode, DTRes *& res, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) {
    ColumnCalc<DTRes, DTDataLhs, DTDataRhs>::apply(opCode, res, data_lhs, data_rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT>
struct ColumnCalc<tuddbs::Column<VT>, tuddbs::Column<VT>, tuddbs::Column<VT>> {
    static void apply(BinaryOpCode opCode, tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data_lhs, const tuddbs::Column<VT> * data_rhs, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, tsl::avx512>;
        if (opCode==BinaryOpCode::ADD) {
            tuddbs::daphne_calc<ps, tsl::functors::add> calc;
            res = calc(data_lhs, data_rhs);   
        } else if (opCode==BinaryOpCode::MUL) {
            tuddbs::daphne_calc<ps, tsl::functors::mul> calc;
            res = calc(data_lhs, data_rhs);
        } else if (opCode==BinaryOpCode::AND) {
            tuddbs::daphne_calc<ps, tsl::functors::binary_and> calc;
            res = calc(data_lhs, data_rhs);
        } else if (opCode==BinaryOpCode::OR) {
            tuddbs::daphne_calc<ps, tsl::functors::binary_or> calc;
            res = calc(data_lhs, data_rhs);
        } else {
            throw std::runtime_error("Unsupported binary operator");
        }
        
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNCALC_H