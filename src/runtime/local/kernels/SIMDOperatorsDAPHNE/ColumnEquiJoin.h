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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNEQUIJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNEQUIJOIN_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/equiJoin.hpp>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTDataLhs, class DTDataRhs>
struct ColumnEquiJoin {
    static void apply(DTRes *& res, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTDataLhs, class DTDataRhs>
void columnEquiJoin(DTRes *& res, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) {
    ColumnEquiJoin<DTRes, DTDataLhs, DTDataRhs>::apply(res, data_lhs, data_rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT>
struct ColumnEquiJoin<tuddbs::Column<VT>, tuddbs::Column<VT>, tuddbs::Column<VT>> {
    static void apply(tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data_lhs, const tuddbs::Column<VT> * data_rhs, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, tsl::avx512>;
        if (data_lhs->getPopulationCount() < data_rhs->getPopulationCount()) {
            res = tuddbs::natural_equi_join<ps>(data_lhs, data_rhs);
        } else {
            res = tuddbs::natural_equi_join<ps>(data_rhs, data_lhs);
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNEQUIJOIN_H