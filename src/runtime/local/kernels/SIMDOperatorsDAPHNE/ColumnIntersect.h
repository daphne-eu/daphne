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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNINTERSECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNINTERSECT_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/intersect.hpp>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTPosLhs, class DTPosRhs>
struct ColumnIntersect {
    static void apply(DTRes *& res, const DTPosLhs * pos_lhs, const DTPosRhs * pos_rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTPosLhs, class DTPosRhs>
void columnIntersect(DTRes *& res, const DTPosLhs * pos_lhs, const DTPosRhs * pos_rhs, DCTX(ctx)) {
    ColumnIntersect<DTRes, DTPosLhs, DTPosRhs>::apply(res, pos_lhs, pos_rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTPos1, typename VTPos2>
struct ColumnIntersect<tuddbs::Column<VTRes>, tuddbs::Column<VTPos1>, tuddbs::Column<VTPos2>> {
    static void apply(tuddbs::Column<VTRes> *& res, const tuddbs::Column<VTPos1> * pos_lhs, const tuddbs::Column<VTPos2> * pos_rhs, DCTX(ctx)) {
        using ps = typename tsl::simd<VTPos1, tsl::avx512>;
        tuddbs::daphne_intersect<ps> intersect;
        res = intersect(pos_lhs, pos_rhs);   
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNINTERSECT_H