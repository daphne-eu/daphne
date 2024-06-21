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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/ContiguousTensor.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <concepts>

template<typename T>
concept Float_t = std::is_floating_point<T>::value;

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, Float_t FloatType>
struct Quantize {
    static void apply(DTRes *& res, const DTArg * arg, FloatType min, FloatType max, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg, Float_t FloatType>
void quantize(DTRes *& res, const DTArg * arg, FloatType min, FloatType max, DCTX(ctx)) {
    Quantize<DTRes, DTArg, FloatType>::apply(res, arg, min, max, ctx);
}

template<Float_t FloatType>
void calc_quantization_params(FloatType min, FloatType max, FloatType& scale, uint8_t& quantized_zero) {
    // Make sure that 0 is included
    min = (min > static_cast<FloatType>(0)) ? static_cast<FloatType>(0) : min;
    max = (max < static_cast<FloatType>(0)) ? static_cast<FloatType>(0) : max;

    const uint8_t q_min = 0;
    const uint8_t q_max = 255;

    scale = (max - min) / static_cast<FloatType>(1 + q_max - q_min);

    FloatType mapped_zero = static_cast<FloatType>(q_max) - max / scale;
    if (mapped_zero < static_cast<FloatType>(q_min)) {
        quantized_zero = static_cast<uint8_t>(q_min);
    }
    else if (mapped_zero > static_cast<FloatType>(q_max)) {
        quantized_zero = static_cast<uint8_t>(q_max);
    }
    else {
        // Rounds half-way cases away from zero.
        quantized_zero = static_cast<uint8_t>((std::roundf(mapped_zero)));
    }
}

template<Float_t FloatType>
uint8_t quantize_value(FloatType a, FloatType scale, uint8_t quantized_zero) {
    // Map
    FloatType value = static_cast<FloatType>(quantized_zero) + a / scale;

    // Clip
    value = (value > static_cast<FloatType>(255)) ? static_cast<FloatType>(255) : value;
    value = (value < static_cast<FloatType>(0)) ? static_cast<FloatType>(0) : value;

    // Round
    return static_cast<uint8_t>((std::roundf(value)));
}

// ****************************************************************************
// Template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<Float_t FloatType>
struct Quantize<DenseMatrix<uint8_t>, DenseMatrix<FloatType>, FloatType> {
    static void apply(DenseMatrix<uint8_t> *& res, const DenseMatrix<FloatType> * arg, FloatType min, FloatType max, DCTX(ctx)) {
        const size_t nr1 = arg->getNumRows();
        const size_t nc1 = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<uint8_t>>(nr1, nc1, false);
        }
        else {
            assert((nr1 == res->getNumRows()) && "#rows of res and #rows of rhs must be the same");
            assert((nc1 == res->getNumCols()) && "#cols of res and #cols of rhs must be the same");
        }

        FloatType scale = 0;
        uint8_t q_zero = 0;
        calc_quantization_params<FloatType>(min, max, scale, q_zero);
        for (int i = 0; i < (int)nr1; i++) {
            for (int j = 0; j < (int)nc1; j++) {
                res->set(i,j, quantize_value<FloatType>(arg->get(i,j), scale, q_zero));
            }
        }
    }
};

// ----------------------------------------------------------------------------
// ContiguousTensor <- ContiguousTensor
// ----------------------------------------------------------------------------

template<Float_t FloatType>
struct Quantize<ContiguousTensor<uint8_t>, ContiguousTensor<FloatType>, FloatType> {
    static void apply(ContiguousTensor<uint8_t> *& res, const ContiguousTensor<FloatType> * arg, FloatType min, FloatType max, DCTX(ctx)) {
        if(res == nullptr) {
            res = DataObjectFactory::create<ContiguousTensor<uint8_t>>(arg->tensor_shape, InitCode::NONE);
        }

        FloatType scale = 0;
        uint8_t q_zero = 0;
        calc_quantization_params<FloatType>(min, max, scale, q_zero);
        for(size_t i=0; i<res->total_element_count; i++) {
            res->data[i] = quantize_value<FloatType>(arg->data[i], scale, q_zero);
        }
    }
};

// ----------------------------------------------------------------------------
// ContiguousTensor <- ContiguousTensor
// ----------------------------------------------------------------------------

template<Float_t FloatType>
struct Quantize<ChunkedTensor<uint8_t>, ChunkedTensor<FloatType>, FloatType> {
    static void apply(ChunkedTensor<uint8_t> *& res, const ChunkedTensor<FloatType> * arg, FloatType min, FloatType max, DCTX(ctx)) {
        if(res == nullptr) {
            res = DataObjectFactory::create<ChunkedTensor<uint8_t>>(arg->tensor_shape, arg->chunk_shape, InitCode::NONE);
        }

        FloatType scale = 0;
        uint8_t q_zero = 0;
        calc_quantization_params<FloatType>(min, max, scale, q_zero);
        for(size_t i=0; i<res->total_size_in_elements; i++) {
            res->data[i] = quantize_value<FloatType>(arg->data[i], scale, q_zero);
        }
    }
};