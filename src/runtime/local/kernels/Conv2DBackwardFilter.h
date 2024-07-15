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

#include <limits>
#include <random>
#include <type_traits>

#include <cstddef>
#include <cstdint>

#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg>
struct Conv2DBackwardFilter
{
    static void apply(  DTRes *&dFilter,
                        const DTArg *input,
                        const DTArg *output,
                        const size_t stride_h, const size_t stride_w,
                        const size_t pad_h, const size_t pad_w,
                        const size_t input_batch_size,
                        const size_t input_num_channels,
                        const size_t input_h, const size_t input_w,
                        const size_t filter_num_filters,
                        const size_t filter_num_channels,
                        const size_t filter_h, const size_t filter_w,
                        DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void conv2DBackwardFilter(DTRes *&dFilter,
                        const DTArg *input,
                        const DTArg *output,
                        const size_t stride_h, const size_t stride_w,
                        const size_t pad_h, const size_t pad_w,
                        const size_t input_batch_size,
                        const size_t input_num_channels,
                        const size_t input_h, const size_t input_w,
                        const size_t filter_num_filters,
                        const size_t filter_num_channels,
                        const size_t filter_h, const size_t filter_w,
                        DCTX(dctx))
{
    Conv2DBackwardFilter<DTRes, DTArg>::apply(
        dFilter, input, output, stride_h, stride_w, pad_h, pad_w,
        input_batch_size, input_num_channels, input_h, input_w,
        filter_num_filters, filter_num_channels, filter_h, filter_w,
        dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
static inline void
GetGradientMatrix(  VT *matrix, const VT *output, 
                    const size_t output_h, const size_t output_w, 
                    const size_t filter_h, const size_t filter_w,
                    const size_t padded_img_h, const size_t padded_img_w,
                    const size_t stride_h, const size_t stride_w, const uint32_t off)
{
    auto matrix_h = padded_img_h - (filter_h - 1);
    auto matrix_w = padded_img_w - (filter_w - 1);

    for (uint32_t i = 0; i < matrix_h * matrix_w; i++)
        matrix[i] = 0;

    auto off_matrix = 0;
    for (uint32_t h = 0, i = 0; h < output_h; h++)
        for (uint32_t w = 0; w < output_w; w++, i++)
        {
            off_matrix = h * stride_h * matrix_w + w * stride_w;
            matrix[off_matrix] = output[off + i];
        }
}

template <typename VT>
static inline void
GetRotatedFilter1(const VT *filter, VT *rotated_filter,
                 size_t filter_h, size_t filter_w, uint32_t off)
{
    for (uint32_t i = 0; i < filter_h * filter_w; i++)
        rotated_filter[i] = filter[off + filter_h * filter_w - i - 1];
}

template <typename VT>
static inline void
Padding6(VT *padded_input, const VT *input, size_t pad_h, size_t pad_w, size_t img_w, size_t img_h, uint32_t off)
{
    auto padded_w = img_w + 2 * pad_w;
    for (uint32_t i = 0; i < img_h * img_w; i++)
        padded_input[i] = 0;
    
    auto start = pad_h * padded_w + pad_w;
    for (uint32_t i = 0, j = 0; i < img_h; i++)
        for (uint32_t k = 0; k < img_w; k++, j++)
            padded_input[start + i * padded_w + k] = input[off + j];
}

uint32_t getPQ2(uint32_t img_extent, uint32_t filter_extent, uint32_t pad_extent, uint32_t stride_extent)
{
    uint32_t padded_image_extent = img_extent + 2 * pad_extent;
    return (padded_image_extent - filter_extent) / stride_extent + 1;
}

template <typename VTRes, typename VTArg>
struct Conv2DBackwardFilter<DenseMatrix<VTRes>, DenseMatrix<VTArg>>
{

    static void
    apply(  DenseMatrix<VTRes> *&dFilter,
            const DenseMatrix<VTArg> *input,
            const DenseMatrix<VTArg> *output,
            const size_t stride_h, const size_t stride_w,
            const size_t pad_h, const size_t pad_w,
            const size_t input_batch_size,
            const size_t input_num_channels,
            const size_t input_h, const size_t input_w,
            const size_t filter_num_filters,
            const size_t filter_num_channels,
            const size_t filter_h, const size_t filter_w,
            DCTX(dctx))
    {
        auto HW = input_h * input_w;
        auto C = input_num_channels;
        auto CHW = C * HW;
        // padded height/width
        auto P = getPQ2(input_h, filter_h, pad_h, stride_h);
        auto Q = getPQ2(input_w, filter_w, pad_w, stride_w);

        auto output_h = P;
        auto output_w = Q;
        auto o_CHW = filter_num_filters * output_h * output_w;
        auto o_HW = output_h * output_w;

        auto start = 0;
        auto stop = input_batch_size;


        auto f_HW = filter_h * filter_w;
        auto f_CHW = filter_num_channels * f_HW;

        auto padded_img_h = input_h + 2 * pad_h;
        auto padded_img_w = input_w + 2 * pad_w;

        auto matrix_h = padded_img_h - (filter_h - 1);
        auto matrix_w = padded_img_w - (filter_w - 1);

        DenseMatrix<VTArg> *gradient_matrix = DataObjectFactory::create<DenseMatrix<VTArg>>(1, matrix_h * matrix_w, true);
        DenseMatrix<VTArg> *padded_matrix = DataObjectFactory::create<DenseMatrix<VTArg>>(1, padded_img_h * padded_img_w, true);
        
        if (dFilter == nullptr)
        {
            dFilter = DataObjectFactory::create<DenseMatrix<VTArg>>(filter_num_filters, filter_num_channels * filter_h * filter_w, true);
        }

        for (uint32_t f = 0; f < filter_num_filters; f++)
        {
            for (uint32_t c = 0; c < filter_num_channels; c++)
            {
                for (uint32_t i = start; i < stop; i++)
                {
                    auto off_input = i * CHW + c * HW;
                    Padding6(padded_matrix->getValues(), input->getValues(), pad_h, pad_w, input_w, input_h, off_input);
                    auto off_output = i * o_CHW + f * o_HW;
                    GetGradientMatrix(gradient_matrix->getValues(), output->getValues(), output_h, output_w, filter_h, filter_w, padded_img_h, padded_img_w, stride_h, stride_w, off_output);

                    for (uint32_t p = 0; p < filter_h; p++)
                        for (uint32_t q = 0; q < filter_w; q++)
                            for (uint32_t h = 0; h < matrix_h; h++)
                                for (uint32_t w = 0; w < matrix_w; w++)
                                    {
                                        auto off_filter = f * f_CHW + c * f_HW + p * filter_h + q;
                                        auto off_matrix = h * matrix_w + w;
                                        auto off_padded = (p + h) * padded_img_w + q + w;
                                        dFilter->getValues()[off_filter] = dFilter->getValues()[off_filter]
                                                                         + padded_matrix->getValues()[off_padded]
                                                                         * gradient_matrix->getValues()[off_matrix];
                                    }
                } 
            }
        }
        DataObjectFactory::destroy(gradient_matrix);
        DataObjectFactory::destroy(padded_matrix);
    }
};