/*
 * Copyright 2024 The DAPHNE Consortium
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

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Tensor.h>
#include <runtime/local/io/io_uring/AsyncUtil.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

struct AsyncIOInfo {
    std::atomic<IO_STATUS> status = IO_STATUS::PRE_SUBMISSION;
    bool needs_byte_reversal      = false;
};

template<typename VT>
void ReverseArray(VT *data, uint64_t element_count) {
    for (uint64_t i = 0; i < element_count; i++) {
        VT tmp = data[i];
        for (uint32_t j = 0; j < sizeof(VT); j++) {
            *(reinterpret_cast<uint8_t *>(&(data[i])) + sizeof(VT) - j) = *(reinterpret_cast<uint8_t *>(&tmp) + j);
        }
    }
}

/**
* @brief A chunked tensor implementation
*
* This tensor implementation is backed by a single array of values. Compared to the ContiguousTensor for a chunked
* tensor, a chunked tensor of rank N with a given shape (its size in each dimension) is further logically split into
* chunks with its own, typically much smaller chunk_shape. In this implementation all chunks have the same chunk_shape.
* Within the chunks the contained elements are arranged in the "higher dimensional equivalent of row-major" order, i.e.
* in the same order as a equivalent ContiguosTensor with shape==chunk_shape.
* The chunks themselves are then also arranged in "row-major" order. An alternative view of this memory layout is to 
* interpret the initially rank N tensor as a 2N-1 rank tensor arranged in "row-major" order (with the "last" N
* dimensions given by the chunk_shape and the first N-1 dimensions reduced in size by the corresponding factor).
*
* The two main advantages of this memory layout are:
* 
* 1. Fine grained and flexible control over the memory layout of the data structure by initially choosing a specific
* chunk_shape or later on choosing to "rechunk" the tensor to a new chunk_shape.
*
* This allows the data structure layout to be adapted to the access pattern of specific kernels making their access
* patterns potentially significantly more cache friendly, which is especially relevant for larger tensors. Typical
* examples here are e.g. image processing kernels operating on 2D slices of a tensor with N >= 2 (smoothing, gradients,
* lvl adjustments, etc.), reduce operations over dimensions that are not the primary dimension (a common example would 
* be an averaging step over the time dimension) or e.g. solving higher dim differential equations over a grid.
*
* 2. Provides a convenient point to sensibly integrate async I/O and ideally also async processing, by loading individual
*    chunks asynchronously and ideally starting to process them asynchronously as well, immediately after a chunk has
*    been read rather than waiting for all chunks to arrive, i.e. block.
*
* While this implementation is already an improvement in respect to memory layout and the options for partial and/or 
* async I/O and/or processing, the choice here to use a single allocation rather than individual allocations per chunk
* limit its use to tensors that fit into memory and is potentially wasteful even for those which do.
*/
template<typename ValueType>
class ChunkedTensor : public Tensor<ValueType> {
    public:
    std::vector<size_t> chunk_shape;
    size_t chunk_element_count;
    std::vector<size_t> chunk_strides;
    std::vector<size_t> intra_chunk_strides;
    std::vector<size_t> chunks_per_dim;
    // In this implementation overhanging chunks are stored as full chunks instead
    // of partial chunks. This also means that total_size_in_elements and
    // total_element_count are not necessarily the same
    size_t total_size_in_elements;
    size_t total_chunk_count;

    std::unique_ptr<std::atomic<bool>[]> chunk_materialization_flags;
    std::unique_ptr<AsyncIOInfo[]> chunk_io_futures;

    std::shared_ptr<ValueType[]> data;

    private:
    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template<class DataType, typename ... ArgTypes>
    friend DataType * DataObjectFactory::create(ArgTypes ...);
    template<class DataType>
    friend void DataObjectFactory::destroy(const DataType * obj);

    ChunkedTensor(const std::vector<size_t> &tensor_shape, const std::vector<size_t> &chunk_shape, InitCode init_code)
        : Tensor<ValueType>::Tensor(tensor_shape), chunk_shape(chunk_shape) {

        chunk_strides.resize(this->rank);
        intra_chunk_strides.resize(this->rank);

        if (this->rank > 0) {
            intra_chunk_strides[0] = 1;
        }

        for(size_t i=0; i<this->rank; i++) {
            if ((tensor_shape[i] == 0) || chunk_shape[i] == 0) {
                throw std::runtime_error("Tensors with dimensions of extent 0 are disallowed.");
            }
        }

        for (size_t i = 1; i < this->rank; i++) {
            intra_chunk_strides[i] = intra_chunk_strides[i - 1] * chunk_shape[i - 1];
        }

        chunk_element_count = this->rank > 0 ? chunk_shape[0] : 1;
        for (size_t i = 1; i < this->rank; i++) {
            chunk_element_count *= chunk_shape[i];
        }

        total_chunk_count = 1;
        chunks_per_dim.resize(this->rank);
        for (size_t i = 0; i < this->rank; i++) {
            chunks_per_dim[i] = this->tensor_shape[i] % chunk_shape[i] == 0
                                  ? this->tensor_shape[i] / chunk_shape[i]
                                  : (this->tensor_shape[i] / chunk_shape[i]) + 1;
            total_chunk_count *= chunks_per_dim[i];
            chunk_strides[i] = i == 0 ? chunk_element_count : chunk_strides[i - 1] * chunks_per_dim[i - 1];
        }

        total_size_in_elements = total_chunk_count * chunk_element_count;

        data = std::shared_ptr<ValueType[]>(new ValueType[total_size_in_elements], std::default_delete<ValueType[]>());

        chunk_materialization_flags = std::make_unique<std::atomic<bool>[]>(total_chunk_count);
        chunk_io_futures            = std::make_unique<AsyncIOInfo[]>(total_chunk_count);
        for (size_t i = 0; i < total_chunk_count; i++) {
            chunk_io_futures[i].needs_byte_reversal = false;
            chunk_io_futures[i].status              = IO_STATUS::PRE_SUBMISSION;
        }

        switch (init_code) {
            case InitCode::NONE:
                for (size_t i = 0; i < total_chunk_count; i++) {
                    chunk_materialization_flags[i] = false;
                }
                break;
            case InitCode::ZERO: {
                for (size_t i = 0; i < total_chunk_count; i++) {
                    chunk_materialization_flags[i] = true;
                }
                for (size_t i = 0; i < total_size_in_elements; i++) {
                    data.get()[i] = 0;
                }
                break;
            }
            case InitCode::MAX: {
                for (size_t i = 0; i < total_chunk_count; i++) {
                    chunk_materialization_flags[i] = true;
                }
                for (size_t i = 0; i < total_size_in_elements; i++) {
                    data.get()[i] = std::numeric_limits<ValueType>::max();
                }
                break;
            }
            case InitCode::MIN: {
                for (size_t i = 0; i < total_chunk_count; i++) {
                    chunk_materialization_flags[i] = true;
                }
                for (size_t i = 0; i < total_size_in_elements; i++) {
                    data.get()[i] = std::numeric_limits<ValueType>::min();
                }
                break;
            }
            case InitCode::IOTA: {
                for (size_t i = 0; i < total_chunk_count; i++) {
                    chunk_materialization_flags[i] = true;
                }
                if (this->rank == 0) {
                    data[0] = 0;
                } else {
                    std::vector<size_t> linear_strides;
                    linear_strides.resize(this->rank);

                    linear_strides[0] = 1;
                    for (size_t i = 1; i < this->rank; i++) {
                        linear_strides[i] = linear_strides[i - 1] * this->tensor_shape[i - 1];
                    }
                    std::vector<size_t> current_ids;
                    current_ids.resize(this->rank);

                    for (size_t i = 0; i < this->total_element_count; i++) {
                        size_t tmp = i;
                        for (int64_t j = this->rank - 1; j >= 0; j--) {
                            current_ids[static_cast<size_t>(j)] = tmp / linear_strides[static_cast<size_t>(j)];
                            tmp                                 = tmp % linear_strides[static_cast<size_t>(j)];
                        }
                        set(current_ids, i);
                    }
                }
                break;
            }
        }
    }

    template<typename VTArg>
    explicit ChunkedTensor(const ChunkedTensor<VTArg> *other)
        : Tensor<ValueType>::Tensor(other->tensor_shape), chunk_shape(other->chunk_shape),
          chunk_element_count(other->chunk_element_count), chunk_strides(other->chunk_strides),
          intra_chunk_strides(other->intra_chunk_strides), chunks_per_dim(other->chunks_per_dim),
          total_size_in_elements(other->total_size_in_elements), total_chunk_count(other->total_chunk_count),
          chunk_materialization_flags(std::make_unique<std::atomic<bool>[]>(total_chunk_count)) {
        data = std::shared_ptr<ValueType[]>(new ValueType[total_size_in_elements], std::default_delete<ValueType[]>());
        for (size_t i = 0; i < total_chunk_count; i++) {
            chunk_materialization_flags[i] = static_cast<bool>(other->chunk_materialization_flags[i]);
        }
        chunk_io_futures = std::make_unique<AsyncIOInfo[]>(total_chunk_count);
        for (size_t i = 0; i < total_chunk_count; i++) {
            chunk_io_futures[i].needs_byte_reversal = other->chunk_io_futures[i].needs_byte_reversal;
            chunk_io_futures[i].status              = other->chunk_io_futures[i].status.load();
        }
        for(size_t i=0; i<total_size_in_elements; i++) {
            data[i] = static_cast<ValueType>(other->data[i]);
        }
    }

    ChunkedTensor(const DenseMatrix<ValueType> *matrix, size_t chunk_size_x, size_t chunk_size_y)
        : Tensor<ValueType>::Tensor(matrix->getNumRows(), matrix->getNumCols()) {
        for(size_t i=0; i<this->rank; i++) {
            if ((this->tensor_shape[i] == 0) || chunk_shape[i] == 0)  {
                throw std::runtime_error("Tensors with dimensions of extent 0 are disallowed.");
            }
        }

        chunk_shape         = {chunk_size_x, chunk_size_y};
        chunk_element_count = chunk_size_x * chunk_size_y;
        chunks_per_dim      = {this->tensor_shape[0] % chunk_size_x == 0 ? this->tensor_shape[0] / chunk_size_x
                                                                         : 1 + (this->tensor_shape[0] / chunk_size_x),
                          this->tensor_shape[1] % chunk_size_y == 0 ? this->tensor_shape[1] / chunk_size_y
                                                                         : 1 + (this->tensor_shape[1] / chunk_size_y)};
        intra_chunk_strides    = {1, chunk_size_x};
        chunk_strides          = {chunk_element_count, chunk_element_count * chunks_per_dim[0]};
        total_chunk_count      = chunks_per_dim[0] * chunks_per_dim[1];
        total_size_in_elements = total_chunk_count * chunk_element_count;

        data = std::shared_ptr<ValueType[]>(new ValueType[total_size_in_elements], std::default_delete<ValueType[]>());

        for (size_t i = 0; i < this->numCols; i++) {
            for (size_t j = 0; j < this->numRows; j++) {
                std::vector<size_t> ids = {i, j};
                set(ids, matrix->get(j, i));
            }
        }

        chunk_materialization_flags = std::make_unique<std::atomic<bool>[]>(total_chunk_count);
        chunk_io_futures            = std::make_unique<AsyncIOInfo[]>(total_chunk_count);
        for (size_t i = 0; i < total_chunk_count; i++) {
            chunk_materialization_flags[i] = true;
            chunk_io_futures[i].needs_byte_reversal = false;
            chunk_io_futures[i].status              = IO_STATUS::PRE_SUBMISSION;
        }
    }

    // Use this + rechunk() for a conversion from a contiguous tensor to a chunked
    // one with arbitrary chunking
    explicit ChunkedTensor(const ContiguousTensor<ValueType> *other)
        : Tensor<ValueType>::Tensor(other->tensor_shape), chunk_shape(other->tensor_shape) {
        chunk_strides.resize(this->rank);
        intra_chunk_strides.resize(this->rank);

        if (this->rank > 0) {
            intra_chunk_strides[0] = 1;
        }

        for(size_t i=0; i<this->rank; i++) {
            if ((this->tensor_shape[i] == 0) || chunk_shape[i] == 0) {
                throw std::runtime_error("Tensors with dimensions of extent 0 are disallowed.");
            }
        }

        for (size_t i = 1; i < this->rank; i++) {
            intra_chunk_strides[i] = intra_chunk_strides[i - 1] * chunk_shape[i - 1];
        }

        chunk_element_count = this->total_element_count;

        total_chunk_count = 1;
        chunks_per_dim.resize(this->rank);
        for (size_t i = 0; i < this->rank; i++) {
            chunks_per_dim[i] = 1;
            total_chunk_count = 1;
            chunk_strides[i]  = chunk_element_count;
        }

        total_size_in_elements = this->total_element_count;

        data = std::shared_ptr<ValueType[]>(new ValueType[total_size_in_elements], std::default_delete<ValueType[]>());

        std::memcpy(data.get(), other->data.get(), total_size_in_elements * sizeof(ValueType));

        chunk_materialization_flags = std::make_unique<std::atomic<bool>[]>(total_chunk_count);
        for (size_t i = 0; i < total_chunk_count; i++) {
            chunk_materialization_flags[i] = true;
        }
        chunk_io_futures = std::make_unique<AsyncIOInfo[]>(total_chunk_count);
        for (size_t i = 0; i < total_chunk_count; i++) {
            chunk_io_futures[i].needs_byte_reversal = false;
            chunk_io_futures[i].status              = IO_STATUS::PRE_SUBMISSION;
        }
    }

    ~ChunkedTensor() override {};

    void printValue(std::ostream &os, ValueType val) const;
    
    public:

    bool operator==(const ChunkedTensor<ValueType> &rhs) const {
        if (this->tensor_shape != rhs.tensor_shape || chunk_shape != rhs.chunk_shape) {
            return false;
        }

        for (size_t i = 0; i < total_chunk_count; i++) {
            if (chunk_materialization_flags[i] != rhs.chunk_materialization_flags[i]) {
                return false;
            }
        }

        std::vector<size_t> current_chunk_id;
        for (size_t i = 0; i < total_chunk_count; i++) {
            if ((!chunk_materialization_flags[i]) || (!rhs.chunk_materialization_flags[i])) {
                return false;
            }

            current_chunk_id = getChunkIdsFromLinearChunkId(i);

            ValueType *lhs_chunk_data = getPtrToChunk(current_chunk_id);
            ValueType *rhs_chunk_data = rhs.getPtrToChunk(current_chunk_id);

            if (lhs_chunk_data != rhs_chunk_data) {
                if (std::memcmp(lhs_chunk_data, rhs_chunk_data, sizeof(ValueType) * chunk_element_count) != 0) {
                    return false;
                }
            }
        }

        return true;
    }

    size_t getLinearId(const std::vector<size_t> &indices) const {
        size_t chunk_id       = indices[0] / chunk_shape[0];
        size_t intra_chunk_id = indices[0] % chunk_shape[0];
        size_t linear_id      = intra_chunk_id + chunk_strides[0] * chunk_id;

        for (size_t i = 1; i < this->rank; i++) {
            chunk_id       = indices[i] / chunk_shape[i];
            intra_chunk_id = indices[i] % chunk_shape[i];
            linear_id += (intra_chunk_id * intra_chunk_strides[i]) + (chunk_id * chunk_strides[i]);
        }
        return linear_id;
    }

    size_t getLinearIdFromChunkIds(const std::vector<size_t> &chunk_indices) const {
        size_t linear_id = chunk_indices[0] * chunk_strides[0];
        for (size_t i = 1; i < this->rank; i++) {
            linear_id += (chunk_indices[i] * chunk_strides[i]);
        }
        return linear_id;
    }

    std::vector<size_t> getChunkIdsFromLinearChunkId(size_t linear_chunk_id) const {
        std::vector<size_t> chunk_ids;
        std::vector<size_t> chunk_id_strides;

        chunk_id_strides.resize(this->rank);
        chunk_id_strides[0] = 1;
        for (size_t i = 1; i < this->rank; i++) {
            chunk_id_strides[i] = (chunks_per_dim[i - 1] * chunk_id_strides[i - 1]);
        }
        
        chunk_ids.resize(this->rank);
        for (int64_t i = this->rank - 1; i >= 0; i--) {
            chunk_ids[i] = linear_chunk_id / chunk_id_strides[i];
            linear_chunk_id = linear_chunk_id % chunk_id_strides[i];
        }

        return chunk_ids;
    }

    size_t getLinearChunkIdFromChunkIds(const std::vector<size_t> &chunk_ids) const {
        if (this->rank == 0) {
            return {};
        }

        std::vector<size_t> chunk_id_strides;

        chunk_id_strides.push_back(1);
        for (size_t i = 1; i < this->rank; i++) {
            chunk_id_strides.push_back(chunks_per_dim[i - 1] * chunk_id_strides[i - 1]);
        }

        size_t linear_chunk_id = 0;
        for (size_t i = 0; i < this->rank; i++) {
            linear_chunk_id += (chunk_ids[i] * chunk_id_strides[i]);
        }

        return linear_chunk_id;
    }

    std::vector<size_t> getChunkIdsFromIds(const std::vector<size_t> &ids) const {
        std::vector<size_t> chunk_ids;

        for (size_t i = 0; i < this->rank; i++) {
            chunk_ids.push_back(ids[i] / chunk_shape[i]);
        }

        return chunk_ids;
    }

    // Ranges inclusive on lower bound and exclusive on upper bound i.e. [x,y] at dsl lvl is in math == [x:y)
    std::optional<std::vector<std::pair<size_t, size_t>>> GetChunkRangeFromIdRange(
      std::vector<std::pair<size_t, size_t>> element_id_ranges) const {
        if (element_id_ranges.size() != this->rank) {
            return std::nullopt;
        }

        // Bounds check ranges
        for (size_t i = 0; i < this->rank; i++) {
            element_id_ranges[i] = {std::get<0>(element_id_ranges[i]), std::get<1>(element_id_ranges[i]) - 1};
            if ((std::get<0>(element_id_ranges[i]) > std::get<1>(element_id_ranges[i])) ||
                (std::get<0>(element_id_ranges[i]) >= this->tensor_shape[i]) ||
                (std::get<1>(element_id_ranges[i]) >= this->tensor_shape[i])) {
                return std::nullopt;
            }
        }

        std::vector<std::pair<size_t, size_t>> chunk_id_ranges;
        chunk_id_ranges.reserve(this->rank);
        for (size_t i = 0; i < this->rank; i++) {
            chunk_id_ranges.push_back({std::get<0>(element_id_ranges[i]) / chunk_shape[i],
                                       1 + (std::get<1>(element_id_ranges[i]) / chunk_shape[i])});
        }
        return chunk_id_ranges;
    }

    // Ranges inclusive on lower bound and exclusive on upper bound i.e. [x,y] at dsl lvl is in math == [x:y)
    std::optional<std::vector<std::vector<size_t>>> GetChunkListFromIdRange (
      std::vector<std::pair<size_t, size_t>> element_id_ranges) const {
        if (element_id_ranges.size() != this->rank) {
            return std::nullopt;
        }
        if (this->rank == 0) {
            return std::nullopt;
        }

        // Bounds check ranges
        for (size_t i = 0; i < this->rank; i++) {
            element_id_ranges[i] = {std::get<0>(element_id_ranges[i]), std::get<1>(element_id_ranges[i]) - 1};
            if ((std::get<0>(element_id_ranges[i]) > std::get<1>(element_id_ranges[i])) ||
                (std::get<0>(element_id_ranges[i]) >= this->tensor_shape[i]) ||
                (std::get<1>(element_id_ranges[i]) >= this->tensor_shape[i])) {
                return std::nullopt;
            }
        }

        std::vector<std::pair<size_t, size_t>> chunk_id_ranges;
        chunk_id_ranges.resize(this->rank);
        for (size_t i = 0; i < this->rank; i++) {
            chunk_id_ranges[i] = {std::get<0>(element_id_ranges[i]) / chunk_shape[i],
                                  std::get<1>(element_id_ranges[i]) / chunk_shape[i]};
        }

        size_t total_chunk_count = std::get<1>(chunk_id_ranges[0]) - std::get<0>(chunk_id_ranges[0]) + 1;
        for (size_t i = 1; i < this->rank; i++) {
            total_chunk_count *= (std::get<1>(chunk_id_ranges[i]) - std::get<0>(chunk_id_ranges[i]) + 1);
        }

        std::vector<size_t> strides;
        strides.push_back(1);
        for (size_t i = 1; i < this->rank; i++) {
            strides.push_back(strides[i - 1] *
                              (std::get<1>(chunk_id_ranges[i - 1]) - std::get<0>(chunk_id_ranges[i - 1]) + 1));
        }

        std::vector<std::vector<size_t>> chunk_id_list;
        chunk_id_list.resize(total_chunk_count);
        for (size_t i = 0; i < total_chunk_count; i++) {
            int64_t tmp = i;
            chunk_id_list[i].resize(this->rank);
            for (int64_t j = (this->rank - 1); j >= 0; j--) {
                chunk_id_list[i][static_cast<size_t>(j)] = std::get<0>(chunk_id_ranges[j]) + (tmp / strides[j]);
                tmp                                      = tmp % strides[j];
            }
        }

        return chunk_id_list;
    }

    // Ranges inclusive on lower bound and exclusive on upper bound i.e. [x,y] at dsl lvl is in math == [x:y)
    std::optional<std::vector<std::vector<size_t>>> GetChunkListFromChunkRange (
      std::vector<std::pair<size_t, size_t>> chunk_id_ranges) const {
        if (chunk_id_ranges.size() != this->rank) {
            return std::nullopt;
        }
        if (this->rank == 0) {
            return std::nullopt;
        }

        // Bounds check ranges
        for (size_t i = 0; i < this->rank; i++) {
            chunk_id_ranges[i] = {std::get<0>(chunk_id_ranges[i]), std::get<1>(chunk_id_ranges[i]) - 1};
            if ((std::get<0>(chunk_id_ranges[i]) > std::get<1>(chunk_id_ranges[i])) ||
                (std::get<0>(chunk_id_ranges[i]) >= this->chunks_per_dim[i]) ||
                (std::get<1>(chunk_id_ranges[i]) >= this->chunks_per_dim[i])) {
                return std::nullopt;
            }
        }

        size_t total_chunk_count = std::get<1>(chunk_id_ranges[0]) - std::get<0>(chunk_id_ranges[0]) + 1;
        for (size_t i = 1; i < this->rank; i++) {
            total_chunk_count *= (std::get<1>(chunk_id_ranges[i]) - std::get<0>(chunk_id_ranges[i]) + 1);
        }

        std::vector<size_t> strides;
        strides.push_back(1);
        for (size_t i = 1; i < this->rank; i++) {
            strides.push_back(strides[i - 1] *
                              (std::get<1>(chunk_id_ranges[i - 1]) - std::get<0>(chunk_id_ranges[i - 1]) + 1));
        }

        std::vector<std::vector<size_t>> chunk_id_list;
        chunk_id_list.resize(total_chunk_count);
        for (size_t i = 0; i < total_chunk_count; i++) {
            int64_t tmp = i;
            chunk_id_list[i].resize(this->rank);
            for (int64_t j = (this->rank - 1); j >= 0; j--) {
                chunk_id_list[i][static_cast<size_t>(j)] = std::get<0>(chunk_id_ranges[j]) + (tmp / strides[j]);
                tmp                                      = tmp % strides[j];
            }
        }

        return chunk_id_list;
    }

    bool IsValueMaterialized(const std::vector<size_t> &indices) const {
        return chunk_materialization_flags[getLinearChunkIdFromChunkIds(getChunkIdsFromIds(indices))];
    }

    bool IsChunkMaterialized(const std::vector<size_t> &chunk_indices) const {
        return chunk_materialization_flags[getLinearChunkIdFromChunkIds(chunk_indices)];
    }

    bool PollChunkMaterializationAndIOStatus(size_t linear_chunk_id) {
        if (chunk_materialization_flags[linear_chunk_id]) {
            return true;
        }

        if (chunk_io_futures[linear_chunk_id].status == IO_STATUS::SUCCESS) {
            if (chunk_io_futures[linear_chunk_id].needs_byte_reversal) {
                ReverseArray(getPtrToChunk(linear_chunk_id), chunk_element_count);
                chunk_io_futures[linear_chunk_id].needs_byte_reversal = false;
            }
            
            chunk_materialization_flags[linear_chunk_id] = true;
            return true;
        }
        return false;
    }

    bool PollChunkMaterializationAndIOStatus(const std::vector<size_t> &chunk_indices) {
        return PollChunkMaterializationAndIOStatus(getLinearChunkIdFromChunkIds(chunk_indices));
    }

    std::optional<ValueType> tryGet(const std::vector<size_t> &indices) const {
        if (indices.size() != this->rank) {
            return std::nullopt;
        }

        if (this->rank == 0) {
            return data.get()[0];
        }

        if (!IsValueMaterialized(indices)) {
            return std::nullopt;
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (indices[i] >= this->tensor_shape[i]) {
                return std::nullopt;
            }
        }
        return data.get()[getLinearId(indices)];
    }

    ValueType get(const std::vector<size_t> &indices) const {
        if (this->rank == 0) {
            return data.get()[0];
        }

        return data.get()[getLinearId(indices)];
    }

    ValueType *tryGetPtrToChunk(const std::vector<size_t> &chunk_indices) const {
        if (chunk_indices.size() != this->rank) {
            return nullptr;
        }

        if (this->rank == 0) {
            return data.get();
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (chunk_indices[i] >= chunks_per_dim[i]) {
                return nullptr;
            }
        }
        return &(data.get()[getLinearIdFromChunkIds(chunk_indices)]);
    }

    ValueType *getPtrToChunk(const std::vector<size_t> &chunk_indices) const {
        if (this->rank == 0) {
            return data.get();
        }
        return &(data.get()[getLinearIdFromChunkIds(chunk_indices)]);
    }

    ValueType *getPtrToChunk(size_t linear_chunk_id) const {
        if (this->rank == 0) {
            return data.get();
        }
        return &(data.get()[linear_chunk_id*chunk_element_count]);
    }

    std::optional<std::unique_ptr<ValueType[]>> tryGetChunk(const std::vector<size_t> &chunk_indices) const {
        if (chunk_indices.size() != this->rank) {
            return std::nullopt;
        }

        if (this->rank == 0) {
            auto result = std::make_unique<ValueType[]>(1);
            result[0]   = data.get()[0];
            return result;
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (chunk_indices[i] >= chunks_per_dim[i]) {
                return std::nullopt;
            }
        }

        if (!IsChunkMaterialized(chunk_indices)) {
            return std::nullopt;
        }

        size_t linear_id = getLinearIdFromChunkIds(chunk_indices);

        auto result = std::make_unique<ValueType[]>(chunk_element_count);
        std::memcpy(result.get(), &(data.get()[linear_id]), chunk_element_count * sizeof(ValueType));
        return result;
    }

    std::unique_ptr<ValueType[]> getChunk(const std::vector<size_t> &chunk_indices) const {
        if (this->rank == 0) {
            auto result = std::make_unique<ValueType[]>(1);
            result[0]   = data.get()[0];
            return result;
        }

        size_t linear_id = getLinearIdFromChunkIds(chunk_indices);

        auto result = std::make_unique<ValueType[]>(chunk_element_count);
        std::memcpy(result.get(), &(data.get()[linear_id]), chunk_element_count * sizeof(ValueType));
        return result;
    }

    bool trySet(const std::vector<size_t> &indices, ValueType value) {
        if (indices.size() != this->rank) {
            return false;
        }
        if (this->rank == 0) {
            data.get()[0] = value;
            return true;
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (indices[i] >= this->tensor_shape[i]) {
                return false;
            }
        }
        data.get()[getLinearId(indices)] = value;

        chunk_materialization_flags[getLinearChunkIdFromChunkIds(getChunkIdsFromIds(indices))] = true;

        return true;
    }

    void set(const std::vector<size_t> &indices, ValueType value) {
        if (this->rank == 0) {
            data.get()[0] = value;
        }
        data.get()[getLinearId(indices)] = value;

        chunk_materialization_flags[getLinearChunkIdFromChunkIds(getChunkIdsFromIds(indices))] = true;
    }

    bool trySetChunk(const std::vector<size_t> &chunk_indices, ValueType *values) {
        if (chunk_indices.size() != this->rank) {
            return false;
        }
        if (this->rank == 0) {
            data.get()[0] = values[0];
            return true;
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (chunk_indices[i] >= this->tensor_shape[i]) {
                return false;
            }
        }

        size_t linear_id = getLinearIdFromChunkIds(chunk_indices);

        std::memcpy(&(data.get()[linear_id]), values, chunk_element_count * sizeof(ValueType));
        chunk_materialization_flags[getLinearChunkIdFromChunkIds(chunk_indices)] = true;
        return true;
    }

    void setChunk(const std::vector<size_t> &chunk_indices, ValueType *values) {
        if (this->rank == 0) {
            data.get()[0]                                                            = values[0];
            chunk_materialization_flags[getLinearChunkIdFromChunkIds(chunk_indices)] = true;
            return;
        }

        size_t linear_id = getLinearIdFromChunkIds(chunk_indices);
        std::memcpy(&(data.get()[linear_id]), values, chunk_element_count * sizeof(ValueType));
        chunk_materialization_flags[getLinearChunkIdFromChunkIds(chunk_indices)] = true;
    }

    bool trySetChunk(const std::vector<size_t> &chunk_indices, const ContiguousTensor<ValueType> &other) {
        if (chunk_indices.size() != this->rank || other.rank != this->rank) {
            return false;
        }

        if (this->rank == 0) {
            data.get()[0]                                                            = other.data.get()[0];
            chunk_materialization_flags[getLinearChunkIdFromChunkIds(chunk_indices)] = true;
            return true;
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (chunk_indices[i] >= this->tensor_shape[i] || other.tensor_shape[i] != chunk_shape[i]) {
                return false;
            }
        }

        size_t linear_id = getLinearIdFromChunkIds(chunk_indices);
        std::memcpy(&(data.get()[linear_id]), other.data.get(), chunk_element_count * sizeof(ValueType));
        chunk_materialization_flags[getLinearChunkIdFromChunkIds(chunk_indices)] = true;

        return true;
    }

    bool tryRechunk(const std::vector<size_t> &new_chunk_shape) {
        if (new_chunk_shape.size() != this->rank) {
            return false;
        }

        // Do not allow rechunking if not all chunks are populated, as this
        // would require materialization to be tracked at
        // entry not chunk lvl
        for (size_t i = 0; i < total_chunk_count; i++) {
            if (!chunk_materialization_flags[i]) {
                return false;
            }
        }

        if (this->rank == 0) {
            return true;
        }

        size_t new_chunk_element_count = new_chunk_shape[0];
        for (size_t i = 1; i < this->rank; i++) {
            new_chunk_element_count *= new_chunk_shape[i];
        }

        std::vector<size_t> new_intra_chunk_strides;
        new_intra_chunk_strides.resize(this->rank);
        new_intra_chunk_strides[0] = 1;
        for (size_t i = 1; i < this->rank; i++) {
            new_intra_chunk_strides[i] = new_intra_chunk_strides[i - 1] * new_chunk_shape[i - 1];
        }

        std::vector<size_t> new_chunk_strides;
        std::vector<size_t> new_chunks_per_dim;
        new_chunk_strides.resize(this->rank);
        new_chunks_per_dim.resize(this->rank);
        size_t new_total_chunk_count = 1;
        for (size_t i = 0; i < this->rank; i++) {
            new_chunks_per_dim[i] = this->tensor_shape[i] % new_chunk_shape[i] == 0
                                      ? this->tensor_shape[i] / new_chunk_shape[i]
                                      : (this->tensor_shape[i] / new_chunk_shape[i]) + 1;
            new_total_chunk_count = new_total_chunk_count * new_chunks_per_dim[i];
            new_chunk_strides[i] =
              i == 0 ? new_chunk_element_count : new_chunk_strides[i - 1] * new_chunks_per_dim[i - 1];
        }

        size_t new_total_size_in_elements = new_total_chunk_count * new_chunk_element_count;

        std::shared_ptr<ValueType[]> new_data(new ValueType[new_total_size_in_elements],
                                              std::default_delete<ValueType[]>());

        std::vector<size_t> chunk_count_strides;
        chunk_count_strides.push_back(1);
        for (size_t i = 1; i < this->rank; i++) {
            chunk_count_strides.push_back(chunk_count_strides[i - 1] * chunks_per_dim[i - 1]);
        }

        std::vector<size_t> current_old_chunk_ids;
        std::vector<size_t> current_old_element_ids;
        current_old_chunk_ids.resize(this->rank);
        current_old_element_ids.resize(this->rank);
        for (size_t i = 0; i < total_chunk_count; i++) {
            size_t tmp = i;
            for (int64_t j = this->rank - 1; j >= 0; j--) {
                current_old_chunk_ids[static_cast<size_t>(j)] = tmp / chunk_count_strides[static_cast<size_t>(j)];
                tmp                                           = tmp % chunk_count_strides[static_cast<size_t>(j)];
            }

            size_t current_chunk_offset = 0;
            for (size_t k = 0; k < this->rank; k++) {
                current_chunk_offset += chunk_strides[k] * current_old_chunk_ids[k];
            }

            for (size_t j = 0; j < chunk_element_count; j++) {
                tmp = j;
                for (int64_t k = this->rank - 1; k >= 0; k--) {
                    current_old_element_ids[static_cast<size_t>(k)] =
                      tmp / intra_chunk_strides[static_cast<size_t>(k)] + current_old_chunk_ids[k] * chunk_shape[k];
                    tmp = tmp % intra_chunk_strides[static_cast<size_t>(k)];
                }

                // Bounds check for partial chunks
                bool out_of_bounds = false;
                for (size_t k = 0; k < this->rank; k++) {
                    if (current_old_element_ids[k] >= this->tensor_shape[k]) {
                        out_of_bounds = true;
                    }
                }

                if (out_of_bounds) {
                    continue;
                }

                size_t new_linear_id = 0;
                for (size_t k = 0; k < this->rank; k++) {
                    size_t current_new_chunk_id       = current_old_element_ids[k] / new_chunk_shape[k];
                    size_t current_new_intra_chunk_id = current_old_element_ids[k] % new_chunk_shape[k];

                    new_linear_id += current_new_chunk_id * new_chunk_strides[k] +
                                     current_new_intra_chunk_id * new_intra_chunk_strides[k];
                }

                new_data[new_linear_id] = data[current_chunk_offset + j];
            }
        }

        chunk_shape            = new_chunk_shape;
        chunk_element_count    = new_chunk_element_count;
        chunk_strides          = new_chunk_strides;
        intra_chunk_strides    = new_intra_chunk_strides;
        chunks_per_dim         = new_chunks_per_dim;
        total_size_in_elements = new_total_size_in_elements;
        total_chunk_count      = new_total_chunk_count;

        data = new_data;

        return true;
    }

    // Ranges inclusive on lower bound and exclusive on upper bound i.e. [x,y] at dsl lvl is in math == [x:y)
    ChunkedTensor<ValueType> *tryDiceAtChunkLvl(std::vector<std::pair<size_t, size_t>>& chunk_ranges) const {
        if (chunk_ranges.size() != this->rank) {
            return nullptr;
        }
        if (this->rank == 0) {
            ChunkedTensor<ValueType> *tmp = DataObjectFactory::create<ChunkedTensor<ValueType>>(
              this->tensor_shape, this->chunk_shape, InitCode::NONE);
            tmp->data.get()[0]             = data.get()[0];
            chunk_materialization_flags[0] = true;
            return tmp;
        }

        for (size_t i = 0; i < this->rank; i++) {
            chunk_ranges[i] = {std::get<0>(chunk_ranges[i]), std::get<1>(chunk_ranges[i]) - 1};
            if (std::get<0>(chunk_ranges[i]) >= chunks_per_dim[i] ||
                std::get<1>(chunk_ranges[i]) >= chunks_per_dim[i] ||
                std::get<0>(chunk_ranges[i]) > std::get<1>(chunk_ranges[i])) {
                return nullptr;
            }
        }

        std::vector<size_t> new_chunk_strides;
        std::vector<size_t> new_chunk_count_per_dim;
        new_chunk_strides.resize(this->rank);
        new_chunk_count_per_dim.resize(this->rank);
        new_chunk_strides[0] = 1;
        new_chunk_count_per_dim[0] = std::get<1>(chunk_ranges[0]) - std::get<0>(chunk_ranges[0]) + 1;
        size_t chunk_count = new_chunk_count_per_dim[0];
        for(size_t i=1; i<this->rank; i++) {
            new_chunk_count_per_dim[i] = std::get<1>(chunk_ranges[i]) - std::get<0>(chunk_ranges[i]) + 1;
            chunk_count *= new_chunk_count_per_dim[i];
            new_chunk_strides[i] = new_chunk_strides[i-1] * new_chunk_count_per_dim[i-1];
        }

        std::vector<size_t> current_chunk_ids;
        current_chunk_ids.resize(this->rank);
        for(size_t i=0; i<chunk_count; i++) {
            size_t tmp = i;
            for(int64_t j=this->rank-1; j>=0; j--) {
                current_chunk_ids[static_cast<size_t>(j)] = (tmp / new_chunk_strides[static_cast<size_t>(j)]) + std::get<0>(chunk_ranges[static_cast<size_t>(j)]);
                tmp = tmp % new_chunk_strides[j];
            }

            if (!IsChunkMaterialized(current_chunk_ids)) {
                return nullptr;
            }
        }

        for (size_t i = 0; i < this->rank; i++) {
            for (size_t j = std::get<0>(chunk_ranges[i]); j <= std::get<1>(chunk_ranges[i]); j++) {
                if (!chunk_materialization_flags[j]) {
                    return nullptr;
                }
            }
        }

        std::vector<size_t> new_tensor_shape;
        new_tensor_shape.resize(this->rank);
        for (size_t i = 0; i < this->rank; i++) {
            new_tensor_shape[i] = (std::get<1>(chunk_ranges[i]) - std::get<0>(chunk_ranges[i]) + 1) * chunk_shape[i];
        }

        ChunkedTensor<ValueType> *new_tensor =
          DataObjectFactory::create<ChunkedTensor<ValueType>>(new_tensor_shape, chunk_shape, InitCode::NONE);

        std::vector<size_t> new_chunk_count_strides;
        new_chunk_count_strides.push_back(1);
        for (size_t i = 1; i < this->rank; i++) {
            new_chunk_count_strides.push_back(new_chunk_count_strides[i - 1] * new_tensor->chunks_per_dim[i - 1]);
        }

        std::vector<size_t> new_current_chunk_ids;
        std::vector<size_t> old_current_chunk_ids;
        new_current_chunk_ids.resize(this->rank);
        old_current_chunk_ids.resize(this->rank);
        for (size_t i = 0; i < new_tensor->total_chunk_count; i++) {
            size_t tmp = i;

            for (int64_t j = this->rank - 1; j >= 0; j--) {
                new_current_chunk_ids[static_cast<size_t>(j)] = tmp / new_chunk_count_strides[static_cast<size_t>(j)];
                old_current_chunk_ids[static_cast<size_t>(j)] =
                  new_current_chunk_ids[static_cast<size_t>(j)] + std::get<0>(chunk_ranges[static_cast<size_t>(j)]);
                tmp = tmp % new_chunk_count_strides[static_cast<size_t>(j)];
            }

            ValueType *ptr_to_old_chunk = getPtrToChunk(old_current_chunk_ids);
            ValueType *ptr_to_new_chunk = new_tensor->getPtrToChunk(new_current_chunk_ids);
            std::memcpy(ptr_to_new_chunk, ptr_to_old_chunk, chunk_element_count * sizeof(ValueType));
            chunk_materialization_flags[getLinearChunkIdFromChunkIds(new_current_chunk_ids)] = true;
        }

        return new_tensor;
    }

    // Ranges inclusive on lower bound and exclusive on upper bound i.e. [x,y] at dsl lvl is in math == [x:y)
    ChunkedTensor<ValueType> *tryDice(std::vector<std::pair<size_t, size_t>> index_ranges,
                                      const std::vector<size_t> &new_chunk_shape) const {
        if (index_ranges.size() != this->rank || new_chunk_shape.size() != this->rank) {
            return nullptr;
        }

        if (this->rank == 0) {
            ChunkedTensor<ValueType> *tmp =
              DataObjectFactory::create<ChunkedTensor<ValueType>>(this->tensor_shape, chunk_shape, InitCode::NONE);
            tmp->data.get()[0] = data.get()[0];
            return tmp;
        }

        for (size_t i = 0; i < this->rank; i++) {
            index_ranges[i] = {std::get<0>(index_ranges[i]), std::get<1>(index_ranges[i]) - 1};
            if (std::get<0>(index_ranges[i]) >= this->tensor_shape[i] ||
                std::get<1>(index_ranges[i]) >= this->tensor_shape[i] ||
                std::get<0>(index_ranges[i]) > std::get<1>(index_ranges[i])) {
                return nullptr;
            }
        }

        std::vector<size_t> new_chunk_strides;
        std::vector<size_t> new_chunk_count_per_dim;
        new_chunk_strides.resize(this->rank);
        new_chunk_count_per_dim.resize(this->rank);
        new_chunk_strides[0] = 1;
        new_chunk_count_per_dim[0] = (std::get<1>(index_ranges[0]) / chunk_shape[0]) - (std::get<0>(index_ranges[0]) / chunk_shape[0]) + 1;
        size_t chunk_count = new_chunk_count_per_dim[0];
        for(size_t i=1; i<this->rank; i++) {
            new_chunk_count_per_dim[i] = (std::get<1>(index_ranges[i]) / chunk_shape[i]) - (std::get<0>(index_ranges[i]) / chunk_shape[i]) + 1;
            chunk_count *= new_chunk_count_per_dim[i];
            new_chunk_strides[i] = new_chunk_strides[i-1] * new_chunk_count_per_dim[i-1];
        }

        std::vector<size_t> current_chunk_ids;
        current_chunk_ids.resize(this->rank);
        for(size_t i=0; i<chunk_count; i++) {
            size_t tmp = i;
            for(int64_t j=this->rank-1; j>=0; j--) {
                current_chunk_ids[static_cast<size_t>(j)] = (tmp / new_chunk_strides[j]) + (std::get<0>(index_ranges[j]) / chunk_shape[j]);
                tmp = tmp % new_chunk_strides[j];
            }

            if (!IsChunkMaterialized(current_chunk_ids)) {
                return nullptr;
            }
        }

        std::vector<size_t> new_tensor_shape;
        new_tensor_shape.resize(this->rank);
        for (size_t i = 0; i < this->rank; i++) {
            new_tensor_shape[i] = std::get<1>(index_ranges[i]) - std::get<0>(index_ranges[i]) + 1;
        }

        ChunkedTensor<ValueType> *new_tensor =
          DataObjectFactory::create<ChunkedTensor<ValueType>>(new_tensor_shape, new_chunk_shape, InitCode::NONE);

        for (size_t i = 0; i < new_tensor->total_chunk_count; i++) {
            new_tensor->chunk_materialization_flags[i] = true;
        }

        std::vector<size_t> new_chunk_count_strides;
        new_chunk_count_strides.push_back(1);
        for (size_t i = 1; i < this->rank; i++) {
            new_chunk_count_strides.push_back(new_chunk_count_strides[i - 1] * new_tensor->chunks_per_dim[i - 1]);
        }

        std::vector<size_t> current_new_chunk_ids;
        std::vector<size_t> current_old_element_ids;
        std::vector<size_t> current_new_element_ids;
        current_new_chunk_ids.resize(this->rank);
        current_old_element_ids.resize(this->rank);
        current_new_element_ids.resize(this->rank);
        for (size_t i = 0; i < new_tensor->total_chunk_count; i++) {
            size_t tmp = i;
            for (int64_t j = this->rank - 1; j >= 0; j--) {
                current_new_chunk_ids[static_cast<size_t>(j)] = tmp / new_chunk_count_strides[static_cast<size_t>(j)];
                tmp                                           = tmp % new_chunk_count_strides[static_cast<size_t>(j)];
            }

            for (size_t j = 0; j < new_tensor->chunk_element_count; j++) {
                tmp = j;
                for (int64_t k = this->rank - 1; k >= 0; k--) {
                    current_new_element_ids[static_cast<size_t>(k)] =
                      tmp / new_tensor->intra_chunk_strides[static_cast<size_t>(k)] +
                      current_new_chunk_ids[k] * new_tensor->chunk_strides[k];
                    tmp = tmp % new_tensor->intra_chunk_strides[static_cast<size_t>(k)];
                }

                // Bounds check for partial chunks
                bool out_of_bounds = false;
                for (size_t k = 0; k < this->rank; k++) {
                    if (current_new_element_ids[k] >= new_tensor->tensor_shape[k]) {
                        out_of_bounds = true;
                    }
                }

                if (out_of_bounds) {
                    continue;
                }

                for (size_t k = 0; k < this->rank; k++) {
                    current_old_element_ids[k] = current_new_element_ids[k] + std::get<0>(index_ranges[k]);
                }

                new_tensor->set(current_new_element_ids, get(current_old_element_ids));
            }
        }
        
        return new_tensor;
    }

    // Ranges inclusive on lower bound and exclusive on upper bound i.e. [x,y] at dsl lvl is in math == [x:y)
    ContiguousTensor<ValueType> *tryDiceToContiguousTensor(std::vector<std::pair<size_t, size_t>> index_ranges) const {
        if (index_ranges.size() != this->rank) {
            return nullptr;
        }

        if (this->rank == 0) {
            ContiguousTensor<ValueType> *tmp =
              DataObjectFactory::create<ContiguousTensor<ValueType>>(this->tensor_shape, InitCode::NONE);
            tmp->data.get()[0] = data.get()[0];
            return tmp;
        }

        for (size_t i = 0; i < this->rank; i++) {
            index_ranges[i] = {std::get<0>(index_ranges[i]), std::get<1>(index_ranges[i]) - 1};
            if (std::get<0>(index_ranges[i]) >= this->tensor_shape[i] ||
                std::get<1>(index_ranges[i]) >= this->tensor_shape[i] ||
                std::get<0>(index_ranges[i]) > std::get<1>(index_ranges[i])) {
                return nullptr;
            }
        }

        std::vector<size_t> new_chunk_strides;
        std::vector<size_t> new_chunk_count_per_dim;
        new_chunk_strides.resize(this->rank);
        new_chunk_count_per_dim.resize(this->rank);
        new_chunk_strides[0] = 1;
        new_chunk_count_per_dim[0] = (std::get<1>(index_ranges[0]) / chunk_shape[0]) - (std::get<0>(index_ranges[0]) / chunk_shape[0]) + 1;
        size_t chunk_count = new_chunk_count_per_dim[0];
        for(size_t i=1; i<this->rank; i++) {
            new_chunk_count_per_dim[i] = (std::get<1>(index_ranges[i]) / chunk_shape[i]) - (std::get<0>(index_ranges[i]) / chunk_shape[i]) + 1;
            chunk_count *= new_chunk_count_per_dim[i];
            new_chunk_strides[i] = new_chunk_strides[i-1] * new_chunk_count_per_dim[i-1];
        }

        std::vector<size_t> current_chunk_ids;
        current_chunk_ids.resize(this->rank);
        for(size_t i=0; i<chunk_count; i++) {
            size_t tmp = i;
            for(int64_t j=this->rank-1; j>=0; j--) {
                current_chunk_ids[static_cast<size_t>(j)] = (tmp / new_chunk_strides[j]) + (std::get<0>(index_ranges[j]) / chunk_shape[j]);
                tmp = tmp % new_chunk_strides[j];
            }

            if (!IsChunkMaterialized(current_chunk_ids)) {
                return nullptr;
            }
        }

        std::vector<size_t> new_tensor_shape;
        new_tensor_shape.resize(this->rank);
        for (size_t i = 0; i < this->rank; i++) {
            new_tensor_shape[i] = std::get<1>(index_ranges[i]) - std::get<0>(index_ranges[i]) + 1;
        }

        ContiguousTensor<ValueType> *new_tensor =
          DataObjectFactory::create<ContiguousTensor<ValueType>>(new_tensor_shape, InitCode::NONE);

        std::vector<size_t> current_new_indices;
        std::vector<size_t> current_old_indices;
        current_new_indices.resize(this->rank);
        current_old_indices.resize(this->rank);
        for (size_t i = 0; i < new_tensor->total_element_count; i++) {
            size_t tmp = i;

            for (int64_t j = this->rank - 1; j >= 0; j--) {
                current_new_indices[static_cast<size_t>(j)] = (tmp / new_tensor->strides[static_cast<size_t>(j)]);
                current_old_indices[static_cast<size_t>(j)] =
                  current_new_indices[static_cast<size_t>(j)] + std::get<0>(index_ranges[static_cast<size_t>(j)]);
                tmp = tmp % new_tensor->strides[static_cast<size_t>(j)];
            }

            new_tensor->set(current_new_indices, get(current_old_indices));
        }

        return new_tensor;
    }

    AsyncIOInfo *GetAsyncIOInfo(const std::vector<size_t> &chunk_ids) {
        return &(chunk_io_futures[getLinearChunkIdFromChunkIds(chunk_ids)]);
    }

    // Prints elements in logical layout
    void print(std::ostream &os) const override {
        os << "ChunkedTensor(";
        for (size_t i = 0; i < this->rank; i++) {
            os << this->tensor_shape[i];
            if (i != this->rank - 1) {
                os << "x";
            }
        }
        os << ", " << ValueTypeUtils::cppNameFor<ValueType> << ")" << std::endl;

        if (this->rank == 0) {
            os << data.get()[0] << std::endl;
            return;
        }

        std::vector<size_t> current_ids;
        current_ids.resize(this->rank);
        std::vector<size_t> linear_strides;
        linear_strides.resize(this->rank);
        linear_strides[0] = 1;
        for (size_t i = 1; i < this->rank; i++) {
            linear_strides[i] = linear_strides[i - 1] * this->tensor_shape[i - 1];
        }
        for (size_t i = 0; i < this->total_element_count; i++) {
            if (i % this->tensor_shape[0] == 0) {
                os << "\n";
            }

            size_t tmp = i;
            for (int64_t j = this->rank - 1; j >= 0; j--) {
                current_ids[static_cast<size_t>(j)] = tmp / linear_strides[static_cast<size_t>(j)];
                tmp                                 = tmp % linear_strides[static_cast<size_t>(j)];
            }

            if (IsValueMaterialized(current_ids)) {
                os << get(current_ids) << " ";
            } else {
                os << "_ ";
            }
        }
        os << std::endl;
    }

    size_t serialize(std::vector<char> &buf) const override {
        throw std::runtime_error("ChunkedTensor::serialize() is not supported (yet)");
    }

    size_t getNumItems() const override {
        return this->total_element_count;
    }
};

template<typename ValueType>
bool areLogicalElementsEqual(const ContiguousTensor<ValueType> &contiguous_tensor,
                             const ChunkedTensor<ValueType> &chunked_tensor) {
    if (contiguous_tensor.tensor_shape != chunked_tensor.tensor_shape) {
        return false;
    }

    for (size_t i = 0; i < chunked_tensor.total_chunk_count; i++) {
        if (!chunked_tensor.chunk_materialization_flags[i]) {
            return false;
        }
    }

    std::vector<size_t> linear_strides;
    linear_strides.resize(chunked_tensor.rank);
    linear_strides[0] = 1;
    for (size_t i = 1; i < chunked_tensor.rank; i++) {
        linear_strides[i] = linear_strides[i - 1] * chunked_tensor.tensor_shape[i - 1];
    }
    std::vector<size_t> current_ids;
    current_ids.resize(chunked_tensor.rank);
    for (size_t i = 0; i < chunked_tensor.total_element_count; i++) {
        size_t tmp = i;
        for (int64_t j = chunked_tensor.rank - 1; j >= 0; j--) {
            current_ids[static_cast<size_t>(j)] = tmp / linear_strides[static_cast<size_t>(j)];
            tmp                                 = tmp % linear_strides[static_cast<size_t>(j)];
        }

        if (contiguous_tensor.get(current_ids) != chunked_tensor.get(current_ids)) {
            return false;
        }
    }

    return true;
}
