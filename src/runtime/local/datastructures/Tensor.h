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

#include <cstddef>
#include <vector>
#include <cstdlib>
#include <stdexcept>

#include <runtime/local/datastructures/Structure.h>

template<typename ValueType>
class Tensor : public Structure {
    public:
    size_t rank;
    std::vector<size_t> tensor_shape;
    size_t total_element_count;

    protected:
    Tensor(const std::vector<size_t>& tensor_shape)
        : Structure(tensor_shape.size() >= 1 ? tensor_shape[0] : 0, tensor_shape.size() >= 2 ? tensor_shape[1] : 0),
          rank(tensor_shape.size()), tensor_shape(tensor_shape) {
        if (rank > 0) {
            total_element_count = tensor_shape[0];
            for (size_t i = 1; i < rank; i++) {
                total_element_count *= tensor_shape[i];
            }
        } else {
            total_element_count = 1;
        }
    };

    Tensor(size_t numRows, size_t numCols) : Structure(numRows, numCols), rank(2) {
        tensor_shape.push_back(numCols);
        tensor_shape.push_back(numRows);
        total_element_count = numRows * numCols;
    };

    virtual ~Tensor() {};

    public:

    virtual size_t getNumDims() const override {
        return rank;
    }
    
    // These pure virtual functions are only well defined for a ND-tensor in the
    // case of N=2. Which dimension is addressed via row and column id is ambiguous
    // for larger N.
    // Use the provided tryDice() function instead.
    virtual Tensor* sliceRow(size_t rl, size_t ru) const override {
        throw std::runtime_error("Tensor::sliceRow() is not supported (yet)");
    }

    virtual Tensor* sliceCol(size_t cl, size_t cu) const override {
        throw std::runtime_error("Tensor::sliceCol() is not supported (yet)");
    }

    virtual Tensor* slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        throw std::runtime_error("Tensor::slice() is not supported (yet)");
    }
};

enum struct InitCode { NONE, ZERO, MAX, MIN, IOTA, RAND };
