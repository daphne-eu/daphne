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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNSELECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNSELECT_H

#include "generated/declarations/compare.hpp"
#include <runtime/local/kernels/SIMDOperatorsDAPHNE/SelectOpCode.h>
#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>
#include <SIMDOperators/datastructure/column.hpp>
#include <SIMDOperators/operators/select.hpp>

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
        std::shared_ptr<const tuddbs::Column<VT> > lhs_shared(lhs);
        std::shared_ptr<const tuddbs::Column<VT> > tmp;
        if (opCode == SelectOpCode::LT) {
            tmp = tuddbs::select<ps, tsl::functors::less_than>::apply(lhs_shared, rhs);
        } else if (opCode == SelectOpCode::LE) {
            tmp = tuddbs::select<ps, tsl::functors::less_than_or_equal>::apply(lhs_shared, rhs);
        } else if (opCode == SelectOpCode::GT) {
            tmp = tuddbs::select<ps, tsl::functors::greater_than>::apply(lhs_shared, rhs);
        } else if (opCode == SelectOpCode::GE) {
            tmp = tuddbs::select<ps, tsl::functors::greater_than_or_equal>::apply(lhs_shared, rhs);
        } else if (opCode == SelectOpCode::EQ) {
            tmp = tuddbs::select<ps, tsl::functors::equal>::apply(lhs_shared, rhs);
        } else if (opCode == SelectOpCode::NEQ) {
            tmp = tuddbs::select<ps, tsl::functors::nequal>::apply(lhs_shared, rhs);
        }
        const tuddbs::Column<VT> * tmp2 = tmp.get();
        tuddbs::Column<VT> * tmp3 = new tuddbs::Column<VT>(*tmp2);
        res = tmp3;
        
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNSELECT_H