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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNSELECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNSELECT_H

#include <runtime/local/kernels/SIMDOperatorsDAPHNE/SelectOpCode.h>
#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/select.hpp>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, typename VTRhs>
struct ColumnSelect {
    static void apply(SelectOpCode opCode, DTRes *& res, const DTLhs * lhs, VTRhs rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, typename VTRhs>
void columnSelect(SelectOpCode opCode, DTRes *& res, const DTLhs * lhs, VTRhs rhs, DCTX(ctx)) {
    ColumnSelect<DTRes, DTLhs, VTRhs>::apply(opCode, res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT>
struct ColumnSelect<tuddbs::Column<VT>, tuddbs::Column<VT>, VT> {
    static void apply(SelectOpCode opCode, tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * lhs, VT rhs, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, tsl::avx512>;

        if (opCode == SelectOpCode::LT) {
            res = tuddbs::daphne_select<ps, tsl::functors::less_than>(lhs, rhs);
        } else if (opCode == SelectOpCode::LE) {
            res = tuddbs::daphne_select<ps, tsl::functors::less_than_or_equal>::apply(lhs, rhs);
        } else if (opCode == SelectOpCode::GT) {
            res = tuddbs::daphne_select<ps, tsl::functors::greater_than>::apply(lhs, rhs);
        } else if (opCode == SelectOpCode::GE) {
            res = tuddbs::daphne_select<ps, tsl::functors::greater_than_or_equal>::apply(lhs, rhs);
        } else if (opCode == SelectOpCode::EQ) {
            res = tuddbs::daphne_select<ps, tsl::functors::equal>::apply(lhs, rhs);
        } else if (opCode == SelectOpCode::NEQ) {
            res = tuddbs::daphne_select<ps, tsl::functors::nequal>::apply(lhs, rhs);
        }
        
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNSELECT_H