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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/ContiguousTensor.h>

#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<typename VT>
struct ReduceRank {
    static void apply(VT* arg, size_t dim_id, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<typename VT>
void reduceRank(VT* arg, size_t dim_id, DCTX(ctx)) {
    return ReduceRank<VT>::apply(arg, dim_id, ctx);
}

template<typename VT>
struct ReduceRank<ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT>* arg, size_t dim_id, DCTX(ctx)) {
        if (arg) {
            if (arg->rank == 0) {
                throw std::runtime_error("Attempted to reduce rank of tensor of rank 0");
            } else if (arg->rank <= dim_id) {
                throw std::runtime_error("Attempted to reduce non existent dimension of tensor");
            } else if (arg->tensor_shape[dim_id] > 1) {
                throw std::runtime_error("Attempted to reduce dim of size > 1. Consider dicing or reducing before.");
            } else {
                arg->rank -= 1;
                arg->tensor_shape.erase(arg->tensor_shape.begin() + dim_id);
                arg->chunk_shape.erase(arg->chunk_shape.begin() + dim_id);
                arg->chunks_per_dim.erase(arg->chunks_per_dim.begin() + dim_id);
                arg->chunk_strides.erase(arg->chunk_strides.begin() + dim_id);
                arg->intra_chunk_strides.erase(arg->intra_chunk_strides.begin() + dim_id);
            }
        }
    }
};

template<typename VT>
struct ReduceRank<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT>* arg, size_t dim_id, DCTX(ctx)) {
        if (arg) {
            if (arg->rank == 0) {
                throw std::runtime_error("Attempted to reduce rank of tensor of rank 0");
            } else if (arg->rank <= dim_id) {
                throw std::runtime_error("Attempted to reduce non existent dimension of tensor");
            } else if (arg->tensor_shape[dim_id] > 1) {
                throw std::runtime_error("Attempted to reduce dim of size > 1. Consider dicing or reducing before.");
            } else {
                arg->rank -= 1;
                arg->tensor_shape.erase(arg->tensor_shape.begin() + dim_id);
                arg->strides.erase(arg->strides.begin() + dim_id);
            }
        }
    }
};
