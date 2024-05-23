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
#include <runtime/local/datastructures/Matrix.h>

#include <stdexcept>

#include <cmath>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Quantize {
    static void apply(DTRes *& res, const DTArg * arg, float min, float max, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void quantize(DTRes *& res, const DTArg * arg, float min, float max, DCTX(ctx)) {
    Quantize<DTRes, DTArg>::apply(res, arg, min, max, ctx);
}

void calc_quantization_params(float min, float max, float& scale, uint8_t& quantized_zero) {
    // Make sure that 0 is included
    min = (min > 0) ? 0 : min;
    max = (max < 0) ? 0 : max;

    const uint8_t q_min = 0;
    const uint8_t q_max = 255;

    scale = (max - min) / (1 + q_max - q_min);

    float mapped_zero = q_max - max/scale;
    if (mapped_zero < q_min) {
        quantized_zero = q_min;
    }
    else if (mapped_zero > q_max) {
        quantized_zero = q_max;
    }
    else {
        // Rounds half-way cases away from zero.
        quantized_zero = (uint8_t)(std::roundf(mapped_zero));
    }
}

uint8_t quantize_value(float a, float scale, uint8_t quantized_zero) {
    // Map
    float value = static_cast<float>(quantized_zero) + a/scale;

    // Clip
    value = (value > 255) ? 255 : value;
    value = (value < 0) ? 0 : value;

    // Round
    return (uint8_t)(std::roundf(value));
}

// ****************************************************************************
// Template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct Quantize<DenseMatrix<uint8_t>, DenseMatrix<float>> {
    static void apply(DenseMatrix<uint8_t> *& res, const DenseMatrix<float> * arg, float min, float max, DCTX(ctx)) {
        const size_t nr1 = arg->getNumRows();
        const size_t nc1 = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<uint8_t>>(nr1, nc1, false);
        } else {
            if (nr1 != res->getNumRows()) {
                throw std::runtime_error("Quantize - #rows of res and #rows of "
                                         "rhs must be the same");
            }
            if (nc1 != res->getNumCols()) {
                throw std::runtime_error("Quantize - #cols of res and #cols of "
                                         "rhs must be the same");
            }
        }

        float scale = 0;
        uint8_t q_zero = 0;
        calc_quantization_params(min, max, scale, q_zero);
        for (int i = 0; i < (int)nr1; i++) {
            for (int j = 0; j < (int)nc1; j++) {
                res->set(i,j, quantize_value(arg->get(i,j), scale, q_zero));
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<>
struct Quantize<Matrix<uint8_t>, Matrix<float>> {
    static void apply(Matrix<uint8_t> *& res, const Matrix<float> * arg, float min, float max, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<uint8_t>>(numRows, numCols, false);
        }
        else if (numRows != res->getNumRows() || numCols != res->getNumCols()) {
            throw std::runtime_error("Quantize: res must have the same shape as arg");
        }

        float scale = 0;
        uint8_t q_zero = 0;
        calc_quantization_params(min, max, scale, q_zero);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r) {
            for (size_t c = 0; c < numCols; ++c) {
                res->append(r, c, quantize_value(arg->get(r, c), scale, q_zero));
            }
        }
        res->finishAppend();
    }
};
