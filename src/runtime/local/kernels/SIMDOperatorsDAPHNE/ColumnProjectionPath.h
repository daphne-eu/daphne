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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNPROJECTIONPATH_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNPROJECTIONPATH_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/projectionPath.hpp>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTData, class DTPos, typename VE>
struct ColumnProjectionPath {
    static void apply(DTRes *& res, const DTData * data, const DTPos ** pos, const size_t number_cols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTData, class DTPos, typename VE>
void columnProjectionPath(DTRes *& res, const DTData * data, const DTPos ** pos, const size_t number_cols, DCTX(ctx)) {
    ColumnProjectionPath<DTRes, DTData, DTPos, VE>::apply(res, data, pos, number_cols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VTData, typename VTPos, typename VE>
struct ColumnProjectionPath<tuddbs::Column<VTData>, tuddbs::Column<VTData>, tuddbs::Column<VTPos>, VE> {
    static void apply(tuddbs::Column<VTData> *& res, const tuddbs::Column<VTData> * data, const tuddbs::Column<VTPos> ** pos, const size_t number_cols, DCTX(ctx)) {
        using ps = typename tsl::simd<VTData, VE>;
        tuddbs::daphne_projection_path<ps> project;
        if (number_cols == 2) {
            const tuddbs::Column<VTData> * pos_cast0 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[0]);
            const tuddbs::Column<VTData> * pos_cast1 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[1]);
            res = project(data, pos_cast0, pos_cast1);
        } else if (number_cols == 3) {
            const tuddbs::Column<VTData> * pos_cast0 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[0]);
            const tuddbs::Column<VTData> * pos_cast1 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[1]);
            const tuddbs::Column<VTData> * pos_cast2 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[2]);
            res = project(data, pos_cast0, pos_cast1, pos_cast2);
        } else if (number_cols == 4) {
            const tuddbs::Column<VTData> * pos_cast0 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[0]);
            const tuddbs::Column<VTData> * pos_cast1 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[1]);
            const tuddbs::Column<VTData> * pos_cast2 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[2]);
            const tuddbs::Column<VTData> * pos_cast3 = reinterpret_cast<const tuddbs::Column<VTData> *>(pos[3]);
            res = project(data, pos_cast0, pos_cast1, pos_cast2, pos_cast3);
        }
        //res = tuddbs::daphne_projection_path<ps>(data, pos);   
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNPROJECTIONPATH_H