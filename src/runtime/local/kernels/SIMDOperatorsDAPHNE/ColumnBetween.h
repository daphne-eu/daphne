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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNBETWEEN_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNBETWEEN_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/between.hpp>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTData, typename VTL, typename VTU, typename VE>
struct ColumnBetween {
    static void apply(DTRes *& res, const DTData * data, const VTL lower_bound, const VTU upper_bound, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTData, typename VTL, typename VTU, typename VE>
void columnBetween(DTRes *& res, const DTData * data, const VTL lower_bound, const VTU upper_bound, DCTX(ctx)) {
    ColumnBetween<DTRes, DTData, VTL, VTU, VE>::apply(res, data, lower_bound, upper_bound, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT, typename VTL, typename VTU, typename VE>
struct ColumnBetween<tuddbs::Column<VT>, tuddbs::Column<VT>, VTL, VTU, VE> {
    static void apply(tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data, const VTL lower_bound, const VTU upper_bound, DCTX(ctx)) {
        using ps = typename tsl::simd<VTL, VE>;
        tuddbs::daphne_between<ps> between;
        res = between(data, lower_bound, upper_bound);   
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNBETWEEN_H