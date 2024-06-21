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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <memory>
#include <optional>

#include "AggOpCode.h"
#include "AggTensorUtil.h"

// This file includes a more general version of the Aggregate kernels given in AggCog.h / AggRow.h etc., as these are
// ambiguous / do not directly apply to any tensor of any rank.
// The dimension to reduce/aggregate over is specified via a mask supplied in agg_dimension_mask.
// Aggregating a tensor of rank N over all dimension will yield a tensor of rank N with tensor_shape[i] == 1 for all i,
// this tensor can then be converted to a cpp scalar type via the reduce rank kernel or the user may call the AggAll
// kernel, which does both operations at once.
// For a version of this kernel that operates on a subset of the tensors data, by additionally supplying ranges see
// AggTensorRange kernels.

// Note: The arg are not const here since the kernels check and update the materialization flags / io status of chunks
// The alternative way of doing this would be to add callbacks that do that to the io engine so the compute kernels do
// not have to modify the arg tensor
// agg_dim_mask passed per ptr since signatures with template args are not handled by the kernels.json parser shrug*

// TODO: IDXmin/max, handle overhanging chunks properly

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<typename VTRes, class DTArg>
struct Agg {
    static VTRes* apply(AggOpCode opCode, bool* agg_dimension_mask, DTArg* arg, DCTX(ctx)) = delete;
};


// ****************************************************************************
// Convenience function
// ****************************************************************************

template<typename VTRes, class DTArg>
VTRes* agg(AggOpCode opCode, bool* agg_dimension_mask, DTArg* arg, DCTX(ctx)) {
    return Agg<VTRes, DTArg>::apply(opCode, agg_dimension_mask, arg, ctx);
}


template<Scalar_t VTRes, Scalar_t VTArg>
struct Agg<ContiguousTensor<VTRes>, ContiguousTensor<VTArg>> {
    static ContiguousTensor<VTRes>* apply(AggOpCode opCode,
                                          bool* agg_dimension_mask,
                                          const ContiguousTensor<VTArg>* arg,
                                          DCTX(ctx)) {
        size_t rank = arg->rank;

        std::vector<size_t> result_tensor_shape = arg->tensor_shape;

        for (size_t i = 0; i < rank; i++) {
            if (agg_dimension_mask[i]) {
                result_tensor_shape[i] = 1;
            }
        }

        ContiguousTensor<VTRes>* result =
          DataObjectFactory::create<ContiguousTensor<VTRes>>(result_tensor_shape, InitCode::NONE);

        AggChunkOPDispatch<VTRes, VTArg, true>(opCode,
                                               result->data.get(),
                                               arg->data.get(),
                                               arg->tensor_shape,
                                               arg->total_element_count,
                                               agg_dimension_mask,
                                               true);

        return result;
    }
};

//  Assumes chunks are either materialized or are async materialized by other thread -> will hang otherwise
template<Scalar_t VTRes, Scalar_t VTArg>
struct Agg<ChunkedTensor<VTRes>, ChunkedTensor<VTArg>> {
    static ChunkedTensor<VTRes>* apply(AggOpCode opCode,
                                       bool* agg_dimension_mask,
                                       ChunkedTensor<VTArg>* arg,
                                       DCTX(ctx)) {
        size_t rank = arg->rank;

        std::vector<size_t> result_tensor_shape = arg->tensor_shape;
        std::vector<size_t> result_chunk_shape  = arg->chunk_shape;
        size_t chunk_count                      = arg->total_chunk_count;

        for (size_t i = 0; i < rank; i++) {
            if (agg_dimension_mask[i]) {
                result_tensor_shape[i] = 1;
                result_chunk_shape[i]  = 1;
            }
        }

        ChunkedTensor<VTRes>* result =
          DataObjectFactory::create<ChunkedTensor<VTRes>>(result_tensor_shape, result_chunk_shape, InitCode::NONE);

        std::vector<bool> dest_chunk_has_been_touched(result->total_chunk_count, false);

        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {    // Sum,Prod,min,max,idxmin,idxmax
            auto chunk_status = std::make_unique<ChunkMap[]>(chunk_count);
            std::vector<size_t> dest_chunk_ids;
            dest_chunk_ids.resize(rank);
            for (size_t i = 0; i < chunk_count; i++) {
                std::vector<size_t> tmp_src_ids = arg->getChunkIdsFromLinearChunkId(i);

                for (size_t j = 0; j < rank; j++) {
                    dest_chunk_ids[j] = agg_dimension_mask[j] ? 0 : tmp_src_ids[j];
                }
                chunk_status[i] = {i, result->getLinearChunkIdFromChunkIds(dest_chunk_ids), true};
            }

            size_t remaining_chunks = chunk_count;
            while (remaining_chunks != 0) {
                for (size_t i = 0; i < chunk_count; i++) {
                    if (chunk_status[i].not_processed_yet) {
                        bool chunk_can_be_proccessed = false;
                        if (arg->chunk_materialization_flags[chunk_status[i].linear_src_id]) {
                            chunk_can_be_proccessed = true;
                        } else {    // chunk not marked materialized, but it may arrived due to async io in the meantime
                            IO_STATUS current_chunk_io_status =
                              arg->chunk_io_futures[chunk_status[i].linear_src_id].status;

                            switch (current_chunk_io_status) {
                                using enum IO_STATUS;
                                case IN_FLIGHT:
                                    break;
                                case SUCCESS:
                                    if (arg->chunk_io_futures[chunk_status[i].linear_src_id].needs_byte_reversal) {
                                        ReverseArray<VTArg>(arg->getPtrToChunk(chunk_status[i].linear_src_id),
                                                            arg->chunk_element_count);
                                        arg->chunk_io_futures[chunk_status[i].linear_src_id].needs_byte_reversal =
                                          false;
                                    }
                                    chunk_can_be_proccessed                                         = true;
                                    arg->chunk_materialization_flags[chunk_status[i].linear_src_id] = true;
                                    break;
                                case PRE_SUBMISSION:
                                    throw std::runtime_error("Chunk required for Aggregate kernel is not materialized and not currently being read from file.");
                                    break;
                                default:
                                    // Error cases like BAD_FD
                                    throw std::runtime_error("Async load of chunk failed");
                                    break;
                            }
                        }

                        if (chunk_can_be_proccessed) {
                            bool is_first_op = !dest_chunk_has_been_touched[chunk_status[i].linear_dest_id];
                            AggChunkOPDispatch<VTRes, VTArg, false>(
                              opCode,
                              result->getPtrToChunk(chunk_status[i].linear_dest_id),
                              arg->getPtrToChunk(chunk_status[i].linear_src_id),
                              arg->chunk_shape,
                              arg->chunk_element_count,
                              agg_dimension_mask,
                              is_first_op);
                            dest_chunk_has_been_touched[chunk_status[i].linear_dest_id] = true;
                            chunk_status[i].not_processed_yet                           = false;
                            remaining_chunks--;
                        }
                    }
                }
            }
        } else {
            // i.e. MEAN and STDDev

            // Restrict mean and stddev to only be applied in one dim (for simplicity and also it seems to me that
            // it is not likely to be sensible to apply it to multiple dims consecutively)
            size_t dims_to_reduce = 0;
            for (size_t i = 0; i < rank; i++) {
                if (agg_dimension_mask[i]) {
                    dims_to_reduce++;
                }
            }
            if (dims_to_reduce > 1) {
                throw std::runtime_error(
                  "Applying mean and stddev over more than one dim at once is currently not supported");
            }

            bool no_aggregation_dim = true;
            for (size_t i = 0; i < rank; i++) {
                if (agg_dimension_mask[i]) {
                    no_aggregation_dim = false;
                    std::vector<std::pair<size_t, size_t>> full_tensor_range;
                    full_tensor_range.resize(rank);
                    for (size_t j = 0; j < rank; j++) {
                        full_tensor_range[j] = {0, arg->tensor_shape[j] + 1};
                    }
                    // Fill the lists
                    std::vector<std::vector<std::vector<size_t>>> current_lists_of_chunks =
                      GetChunkAggregationLists(arg->GetChunkListFromIdRange(full_tensor_range).value(), i);

                    std::vector<bool> chunk_list_fully_arrived(current_lists_of_chunks.size(), false);
                    size_t remaining_lists_to_proccess = current_lists_of_chunks.size();

                    while (remaining_lists_to_proccess != 0) {
                        std::optional<size_t> fully_matrialized_chunk_list_id =
                          CheckChunkListArrival(current_lists_of_chunks, chunk_list_fully_arrived, arg);

                        if (fully_matrialized_chunk_list_id) {
                            std::vector<size_t> dest_chunk_id =
                              current_lists_of_chunks[fully_matrialized_chunk_list_id.value()][0];
                            dest_chunk_id[i]            = 0;
                            size_t linear_dest_chunk_id = result->getLinearChunkIdFromChunkIds(dest_chunk_id);

                            std::vector<VTArg*> src_chunk_id_ptrs;
                            src_chunk_id_ptrs.resize(
                              current_lists_of_chunks[fully_matrialized_chunk_list_id.value()].size());
                            for (size_t j = 0; j < src_chunk_id_ptrs.size(); j++) {
                                src_chunk_id_ptrs[j] = arg->getPtrToChunk(
                                  current_lists_of_chunks[fully_matrialized_chunk_list_id.value()][j]);
                            }

                            AggChunkList<VTRes, VTArg>(result->getPtrToChunk(dest_chunk_id),
                                                       src_chunk_id_ptrs,
                                                       i,
                                                       arg->chunk_shape,
                                                       arg->chunk_element_count,
                                                       opCode);
                            dest_chunk_has_been_touched[linear_dest_chunk_id] = true;
                            remaining_lists_to_proccess--;
                        }
                    }
                }
            }
            if (no_aggregation_dim) {    // for correctness no need to be efficient
                for (size_t i = 0; i < chunk_count; i++) {
                    while (!arg->IsChunkMaterialized(arg->getChunkIdsFromLinearChunkId(i))) {
                    };
                    VTRes* dest = result->getPtrToChunk(result->getChunkIdsFromLinearChunkId(i));
                    VTRes* src  = arg->getPtrToChunk(arg->getChunkIdsFromLinearChunkId(i));
                    for (size_t j = 0; j < arg->chunk_element_count; j++) {
                        dest[j] = static_cast<VTRes>(src[j]);
                    }
                    dest_chunk_has_been_touched[i] = true;
                }
            }
        }

        for (size_t i = 0; i < result->total_chunk_count; i++) {
            if (dest_chunk_has_been_touched[i]) {
                result->chunk_materialization_flags[i] = true;
            }
        }

        return result;
    }
};
