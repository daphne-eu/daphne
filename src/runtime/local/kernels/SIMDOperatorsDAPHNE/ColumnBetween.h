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

template<class DTRes, class DTData, typename VT>
struct ColumnBetween {
    static void apply(DTRes *& res, const DTData * data, const VT lower_bound, const VT higher_bound, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTData, typename VT>
void columnBetween(DTRes *& res, const DTData * data, const VT lower_bound, const VT higher_bound, DCTX(ctx)) {
    ColumnBetween<DTRes, DTData, VT>::apply(res, data, lower_bound, higher_bound, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT>
struct ColumnBetween<tuddbs::Column<VT>, tuddbs::Column<VT>, VT> {
    static void apply(tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data, const VT lower_bound, const VT higher_bound, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, tsl::avx512>;
        res = tuddbs::daphne_between<ps>(data, lower_bound, higher_bound);   
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNBETWEEN_H