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
#include <SIMDOperators/datastructure/column.hpp>
#include <SIMDOperators/operators/project.hpp>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTData, class DTPos>
struct ColumnProject {
    static void apply(DTRes *& res, const DTData * data, const DTPos * pos, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTData, class DTPos>
void columnProject(DTRes *& res, const DTData * data, const DTPos * pos, DCTX(ctx)) {
    ColumnProject<DTRes, DTData, DTPos>::apply(res, data, pos, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT>
struct ColumnProject<tuddbs::Column<VT>, tuddbs::Column<VT>, tuddbs::Column<VT>> {
    static void apply(tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data, const tuddbs::Column<VT> * pos, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, tsl::avx512>;
        std::shared_ptr<const tuddbs::Column<VT> > data_shared(data);
        std::shared_ptr<const tuddbs::Column<VT> > pos_shared(pos);
        std::shared_ptr<const tuddbs::Column<VT> > tmp;
        tmp = tuddbs::project<ps>::apply(data_shared, pos_shared);
        const tuddbs::Column<VT> * tmp2 = tmp.get();
        tuddbs::Column<VT> * tmp3 = new tuddbs::Column<VT>(*tmp2);
        res = tmp3;        
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNPROJECT_H