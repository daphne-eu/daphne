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

template<class DTResLhs, class DTResRhs, class DTDataLhs, class DTDataRhs, typename VE>
struct ColumnEquiJoin {
    static void apply(DTResLhs *& res_lhs, DTResRhs *& res_rhs, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTResLhs, class DTResRhs, class DTDataLhs, class DTDataRhs>
void columnEquiJoin(DTResLhs *& res_lhs, DTResRhs *& res_rhs, const DTDataLhs * data_lhs, const DTDataRhs * data_rhs, DCTX(ctx)) {
    switch (ctx->getUserConfig().vector_extension) {
        case VectorExtensions::AVX512:
            ColumnEquiJoin<DTResLhs, DTResRhs, DTDataLhs, DTDataRhs, tsl::avx512>::apply(res_lhs, res_rhs, data_lhs, data_rhs, ctx);
            break;
        //case VectorExtensions::AVX2:
        //    ColumnEquiJoin<DTResLhs, DTResRhs, DTDataLhs, DTDataRhs, tsl::avx2>::apply(res_lhs, res_rhs, data_lhs, data_rhs, ctx);
        //    break;
        //case VectorExtensions::SSE:
        //    ColumnEquiJoin<DTResLhs, DTResRhs, DTDataLhs, DTDataRhs, tsl::sse>::apply(res_lhs, res_rhs, data_lhs, data_rhs, ctx);
        //    break;
        case VectorExtensions::SCALAR:
            ColumnEquiJoin<DTResLhs, DTResRhs, DTDataLhs, DTDataRhs, tsl::scalar>::apply(res_lhs, res_rhs, data_lhs, data_rhs, ctx);
            break;
        default:
            throw std::runtime_error("Unknown vector extension");
    }
    //ColumnEquiJoin<DTResLhs, DTResRhs, DTDataLhs, DTDataRhs>::apply(res_lhs, res_rhs, data_lhs, data_rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VTRes, typename VTLhs, typename VTRhs, typename VE>
struct ColumnEquiJoin<tuddbs::Column<VTRes>, tuddbs::Column<VTRes>, tuddbs::Column<VTLhs>, tuddbs::Column<VTRhs>, VE> {
    static void apply(tuddbs::Column<VTRes> *& res_lhs, tuddbs::Column<VTRes> *& res_rhs, const tuddbs::Column<VTLhs> * data_lhs, const tuddbs::Column<VTRhs> * data_rhs, DCTX(ctx)) {
        using ps = typename tsl::simd<VTLhs, VE>;
        if (data_lhs->getPopulationCount() < data_rhs->getPopulationCount()) {
            auto res = tuddbs::natural_equi_join<ps>(data_lhs, data_rhs);
            res_lhs = reinterpret_cast<tuddbs::Column<VTRes> *>(std::get<0>(res));
            res_rhs = reinterpret_cast<tuddbs::Column<VTRes> *>(std::get<1>(res));
        } else {
            auto res = tuddbs::natural_equi_join<ps>(data_rhs, data_lhs);
            res_lhs = reinterpret_cast<tuddbs::Column<VTRes> *>(std::get<1>(res));
            res_rhs = reinterpret_cast<tuddbs::Column<VTRes> *>(std::get<0>(res));
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNEQUIJOIN_H