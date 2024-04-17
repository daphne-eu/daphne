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

#include <memory>
#include <cstring>
#include <ostream>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <type_traits>
#include <vector>
#include <optional>
#include <limits>

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Tensor.h>
#include <runtime/local/datastructures/DenseMatrix.h>

template<typename ValueType>
class ContiguousTensor : public Tensor<ValueType> {
    public:
    std::vector<size_t> strides;

    std::shared_ptr<ValueType[]> data;

    ContiguousTensor(const std::vector<size_t> &tensor_shape, InitCode init_code)
        : Tensor<ValueType>::Tensor(tensor_shape),
          data(new ValueType[this->total_element_count], std::default_delete<ValueType[]>()) {
        strides.resize(this->rank);
        if (this->rank > 0) {
            strides[0] = 1;
        }

        for(size_t i=0; i<this->rank; i++) {
            if (tensor_shape[i] == 0) {
                throw std::runtime_error("Tensors with dimensions of extend 0 are disallowed.");
            }
        }

        for (size_t i = 1; i < this->rank; i++) {
            strides[i] = strides[i - 1] * this->tensor_shape[i - 1];
        }

        // No C++20 sigh*
        // data = std::make_shared<ValueType[]>(this->total_element_count);

        switch (init_code) {
            case InitCode::NONE: {
            } break;
            case InitCode::ZERO: {
                for (size_t i = 0; i < this->total_element_count; i++) {
                    data.get()[i] = 0;
                }
                break;
            }
            case InitCode::MAX: {
                for (size_t i = 0; i < this->total_element_count; i++) {
                    data.get()[i] = std::numeric_limits<ValueType>::max();
                }
                break;
            }
            case InitCode::MIN: {
                for (size_t i = 0; i < this->total_element_count; i++) {
                    data.get()[i] = std::numeric_limits<ValueType>::min();
                }
                break;
            }
            case InitCode::IOTA: {
                for (size_t i = 0; i < this->total_element_count; i++) {
                    data.get()[i] = i;
                }
                break;
            }
        }
    };
    
    template<typename VTArg>
    ContiguousTensor(const ContiguousTensor<VTArg> *other)
        : Tensor<ValueType>::Tensor(other->tensor_shape), strides(other->strides) {
        // workarround for old versions of gcc with template specialization bug
        //https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282
        if constexpr (std::is_same<VTArg, ValueType>::value) { 
            data = other->data;
        } else {
            data = std::shared_ptr<ValueType[]>(new ValueType[this->total_element_count], std::default_delete<ValueType[]>());
            //data = std::make_shared<ValueType[]>(this->total_element_count);
         for(size_t i=0; i<this->total_element_count; i++) {
            data[i] = static_cast<ValueType>(other->data[i]);
         }
        }
    };

    ContiguousTensor(const DenseMatrix<ValueType> *other)
        : Tensor<ValueType>::Tensor(other->getNumRows(), other->getNumCols()), data(other->getValuesSharedPtr()) {
        strides = {1, other->getNumCols()};
        for(size_t i=0; i<this->rank; i++) {
            if (this->tensor_shape[i] == 0) {
                throw std::runtime_error("Tensors with dimensions of extend 0 are disallowed.");
            }
        }
    }

    // Copies passed data
    ContiguousTensor(ValueType *input_data, const std::vector<size_t> &tensor_shape)
        : Tensor<ValueType>::Tensor(tensor_shape),
          data(new ValueType[this->total_element_count], std::default_delete<ValueType[]>()) {
        strides.resize(this->rank);
        if (this->rank > 0) {
            strides[0] = 1;
        }

        for(size_t i=0; i<this->rank; i++) {
            if (this->tensor_shape[i] == 0) {
                throw std::runtime_error("Tensors with dimensions of extend 0 are disallowed.");
            }
        }

        for (size_t i = 1; i < this->rank; i++) {
            strides[i] = strides[i - 1] * this->tensor_shape[i - 1];
        }

        std::memcpy(data.get(), input_data, this->total_element_count * sizeof(ValueType));
    }

    ContiguousTensor(std::shared_ptr<ValueType[]>& input_data, const std::vector<size_t> &tensor_shape)
        : Tensor<ValueType>::Tensor(tensor_shape) {
        data = input_data;
        strides.resize(this->rank);
        if (this->rank > 0) {
            strides[0] = 1;
        }

        for(size_t i=0; i<this->rank; i++) {
            if (this->tensor_shape[i] == 0) {
                throw std::runtime_error("Tensors with dimensions of extend 0 are disallowed.");
            }
        }

        for (size_t i = 1; i < this->rank; i++) {
            strides[i] = strides[i - 1] * this->tensor_shape[i - 1];
        }
    }

    // Takes ownership of data
    ContiguousTensor(std::unique_ptr<ValueType[]> input_data, const std::vector<size_t> &tensor_shape)
        : Tensor<ValueType>::Tensor(tensor_shape), data(std::move(input_data)) {
        strides.resize(this->rank);
        if (this->rank > 0) {
            strides[0] = 1;
        }

        for(size_t i=0; i<this->rank; i++) {
            if (this->tensor_shape[i] == 0) {
                throw std::runtime_error("Tensors with dimensions of extend 0 are disallowed.");
            }
        }

        for (size_t i = 1; i < this->rank; i++) {
            strides[i] = strides[i - 1] * this->tensor_shape[i - 1];
        }
    }

    ~ContiguousTensor() override = default;

    void printValue(std::ostream &os, ValueType val) const;

    bool operator==(const ContiguousTensor<ValueType> &rhs) const {
        if (this->tensor_shape != rhs.tensor_shape) {
            return false;
        }

        return !static_cast<bool>(
          std::memcmp(data.get(), rhs.data.get(), this->total_element_count * sizeof(ValueType)));
    }

    DenseMatrix<ValueType> *tryToGetDenseMatrix() const {
        if (this->rank != 2) {
            return nullptr;
        }

        return DataObjectFactory::create<DenseMatrix<ValueType>>(this->tensor_shape[1], this->tensor_shape[0], data);
    }

    std::optional<ValueType> tryGet(const std::vector<size_t> &element_indices) const {
        if (element_indices.size() != this->rank) {
            return std::nullopt;
        }

        if (this->rank == 0) {
            return data.get()[0];
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (element_indices[i] >= this->tensor_shape[i]) {
                return std::nullopt;
            }
        }

        size_t linear_id = element_indices[0];
        for (size_t i = 1; i < this->rank; i++) {
            linear_id += element_indices[i] * strides[i];
        }

        return data.get()[linear_id];
    }

    ValueType get(const std::vector<size_t> &element_indices) const {
        if (this->rank == 0) {
            return data.get()[0];
        }

        size_t linear_id = element_indices[0];
        for (size_t i = 1; i < this->rank; i++) {
            linear_id += element_indices[i] * strides[i];
        }
        return data.get()[linear_id];
    }

    bool trySet(const std::vector<size_t> &element_indices, ValueType value) {
        if (element_indices.size() != this->rank || this->rank == 0) {
            return false;
        }

        for (size_t i = 0; i < this->rank; i++) {
            if (element_indices[i] >= this->tensor_shape[i]) {
                return false;
            }
        }

        size_t linear_id = element_indices[0];
        for (size_t i = 1; i < this->rank; i++) {
            linear_id += element_indices[i] * strides[i];
        }
        data.get()[linear_id] = value;

        return true;
    }

    void set(const std::vector<size_t> &element_indices, ValueType value) {
        size_t linear_id = element_indices[0];
        for (size_t i = 1; i < this->rank; i++) {
            linear_id += element_indices[i] * strides[i];
        }
        data.get()[linear_id] = value;
    }

    void print(std::ostream &os) const override {
        os << "ContiguousTensor(";
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

        for (size_t i = 0; i < this->total_element_count; i++) {
            if (i % this->tensor_shape[0] == 0) {
                os << "\n";
            }
            printValue(os, data.get()[i]);
            os << " ";
        }
        os << std::endl;
    }

    // Ranges inclusive on lower bound and exclusive on upper bound i.e. [x,y] at dsl lvl is in math == [x:y)
    ContiguousTensor<ValueType> *tryDice(std::vector<std::pair<size_t, size_t>> index_ranges) const {
        if (index_ranges.size() != this->rank) {
            return nullptr;
        }
        
        for (size_t i = 0; i < this->rank; i++) {
            index_ranges[i] = {std::get<0>(index_ranges[i]), std::get<1>(index_ranges[i]) - 1};
            if (std::get<0>(index_ranges[i]) >= this->tensor_shape[i] ||
                std::get<1>(index_ranges[i]) >= this->tensor_shape[i] ||
                std::get<0>(index_ranges[i]) > std::get<1>(index_ranges[i])) {
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

        std::vector<size_t> current_indices;
        current_indices.resize(this->rank);
        for (size_t i = 0; i < new_tensor->total_element_count; i++) {
            size_t tmp = i;

            for (int64_t j = this->rank - 1; j >= 0; j--) {
                current_indices[static_cast<size_t>(j)] = (tmp / new_tensor->strides[static_cast<size_t>(j)]) +
                                                          std::get<0>(index_ranges[static_cast<size_t>(j)]);
                tmp = tmp % new_tensor->strides[static_cast<size_t>(j)];
            }
            new_tensor->data[i] = get(current_indices);
        }

        return new_tensor;
    }

    // Removes all dimensions with a size of 1
    void reduceRank() {
        for (size_t i = 0; i < this->rank; i++) {
            if (this->tensor_shape[i] == 1) {
                this->tensor_shape.erase(this->tensor_shape.begin() + i);
                strides.erase(strides.begin() + i);
            }
        }
        this->rank = this->tensor_shape.size();
    }

    size_t getNumItems() const override {
        return this->total_element_count;
    }

    size_t serialize(std::vector<char> &buf) const override {
        // TODO
        return 0;
    }
};
