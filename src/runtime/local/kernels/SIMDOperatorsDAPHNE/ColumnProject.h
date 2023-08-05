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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNPROJECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNPROJECT_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/project.hpp>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTData, class DTPos, typename VE>
struct ColumnProject {
    static void apply(DTRes *& res, const DTData * data, const DTPos * pos, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTData, class DTPos>
void columnProject(DTRes *& res, const DTData * data, const DTPos * pos, DCTX(ctx)) {
    switch (ctx->getUserConfig().vector_extension) {
        case VectorExtensions::AVX512:
            ColumnProject<DTRes, DTData, DTPos, tsl::avx512>::apply(res, data, pos, ctx);
            break;
        //case VectorExtensions::AVX2:
        //    ColumnProject<DTRes, DTData, DTPos, tsl::avx2>::apply(res, data, pos, ctx);
        //    break;
        //case VectorExtensions::SSE:
        //    ColumnProject<DTRes, DTData, DTPos, tsl::sse>::apply(res, data, pos, ctx);
        //    break;
        case VectorExtensions::SCALAR:
            ColumnProject<DTRes, DTData, DTPos, tsl::scalar>::apply(res, data, pos, ctx);
            break;
        default:
            throw std::runtime_error("Unknown vector extension");
    }
    //ColumnProject<DTRes, DTData, DTPos>::apply(res, data, pos, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT, typename VTPos, typename VE>
struct ColumnProject<tuddbs::Column<VT>, tuddbs::Column<VT>, tuddbs::Column<VTPos>, VE> {
    static void apply(tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data, const tuddbs::Column<VTPos> * pos, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, VE>;
        tuddbs::daphne_project<ps> project;
        const tuddbs::Column<VT> * pos_cast = reinterpret_cast<const tuddbs::Column<VT> *>(pos);
        res = project(data, pos_cast);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNPROJECT_H