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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNAGGREGATE_H
#define SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNAGGREGATE_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/wrappers/DAPHNE/aggregate.hpp>
#include <runtime/local/kernels/AggOpCode.h>

#include <cassert>
#include <cstddef>
#include <map>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTData>
struct ColumnAggregate {
    static void apply(AggOpCode opCode, DTRes *& res, const DTData * data, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTData>
void columnAgg(AggOpCode opCode, DTRes *& res, const DTData * data, DCTX(ctx)) {
    ColumnAggregate<DTRes, DTData>::apply(opCode, res, data, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column
// ----------------------------------------------------------------------------

template<typename VT>
struct ColumnAggregate<tuddbs::Column<VT>, tuddbs::Column<VT>> {
    static void apply(AggOpCode opCode, tuddbs::Column<VT> *& res, const tuddbs::Column<VT> * data, DCTX(ctx)) {
        using ps = typename tsl::simd<VT, tsl::avx512>;
        if (opCode==AggOpCode::SUM) {
            tuddbs::daphne_aggregate<ps, tsl::functors::add, tsl::functors::hadd> aggregate;
            res = aggregate(data);   
        } else {
            assert(false);
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SIMDOPERATORSDAPHNE_COLUMNAGGREGATE_H