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

#include <cmath>
#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct Conv2DForward {
    static void apply(DTRes *&res, size_t& res_h, size_t& res_w,
                          const DTArg *data, const DTArg *filter, const DTArg *bias,
                          const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w,
                          const size_t filter_h, const size_t filter_w, 
                          const size_t stride_h, const size_t stride_w, 
                          const size_t pad_h, const size_t pad_w,  DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void conv2DForward(DTRes *&res, size_t& res_h, size_t& res_w,
                        const DTArg *data, const DTArg *filter, const DTArg *bias,
                        const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w,
                        const size_t filter_h, const size_t filter_w, 
                        const size_t stride_h, const size_t stride_w, 
                        const size_t pad_h, const size_t pad_w, DCTX(dctx)) {
    Conv2DForward<DTRes, DTArg>::apply(res, res_h, res_w,
                        data, filter, bias, batch_size, num_channels, img_h, img_w,
                        filter_h, filter_w, 
                        stride_h, stride_w, 
                        pad_h, pad_w, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
static inline VT
ConvOp(VT initial_value, const VT *in, const VT *filter, uint32_t in_start, uint32_t filter_start, uint32_t length)
{
    VT ret = 0;
    for (uint32_t i = 0; i < length; ++i)
        ret += in[in_start + i] * filter[filter_start + i];
    return ret + initial_value;
}

template <typename VT>
static inline void
GetPaddedData(const VT *data, VT *padded_data, VT *selected_data,
              size_t pad_w, size_t pad_h,
              size_t img_w, size_t img_h, size_t padded_img_w, uint32_t off)
{
    uint32_t j = 0;
    uint32_t k = 0;
    uint32_t padded_index = 0;
    uint32_t data_index = 0;
    for (j = 0; j < img_h * img_w; j++)
        selected_data[j] = data[off + j];

    for (j = 0; j < (pad_h * padded_img_w); j++, padded_index++)
        padded_data[padded_index] = 0;
    for (j = 0; j < img_h; j++)
    {
        for (k = 0; k < pad_w; k++, padded_index++)
            padded_data[padded_index] = 0;
        for (k = 0; k < img_w; k++, data_index++, padded_index++)
            padded_data[padded_index] = selected_data[data_index];
        for (k = 0; k < pad_w; k++, padded_index++)
            padded_data[padded_index] = 0;
    }
    for (j = 0; j < (pad_h * padded_img_w); j++, padded_index++)
        padded_data[padded_index] = 0;
}

uint32_t getPQ4(uint32_t img_extent, uint32_t filter_extent, uint32_t pad_extent, uint32_t stride_extent) {
        uint32_t padded_image_extent = img_extent + 2 * pad_extent;
        // float result = static_cast<float>(padded_image_extent - filter_extent) / stride_extent + 1;
        // std::cout<< std::ceil(result)<<std::endl;
        // return std::ceil(result);
        return (padded_image_extent - filter_extent) / stride_extent + 1;
    }


template <typename VTRes, typename VTArg>
struct Conv2DForward<DenseMatrix<VTRes>, DenseMatrix<VTArg>>
{
    static void 
    apply(DenseMatrix<VTRes> *&res, 
          size_t &res_h, size_t &res_w,
          const DenseMatrix<VTArg> *data,
          const DenseMatrix<VTArg> *filter, 
          const DenseMatrix<VTArg> *bias,
          const size_t batch_size, const size_t num_channels, 
          const size_t img_h, const size_t img_w,     
          const size_t filter_h, const size_t filter_w,
          const size_t stride_h, const size_t stride_w,
          const size_t pad_h, const size_t pad_w,  
          DCTX(dctx))
    {
        auto HW = img_h * img_w;
        auto C = num_channels;        
        auto CHW = C * HW;
        // padded height/width
        auto P = getPQ4(img_h, filter_h, pad_h, stride_h);
        auto Q = getPQ4(img_w, filter_w, pad_w, stride_w);
        auto C_new = filter->getNumRows();
        auto PQ = P * Q;
        auto CPQ = C_new * PQ;
        res_h = P;
        res_w = Q;

        auto start = 0;
        auto stop = batch_size;

        auto ii = start * CHW;
        auto oi = start * CPQ;

        auto off_f = 0;
        auto f_CHW = C * filter_h * filter_w;

        auto padded_img_h = img_h + 2 * pad_h;
        auto padded_img_w = img_w + 2 * pad_w;
        DenseMatrix<VTArg> *padded_data = DataObjectFactory::create<DenseMatrix<VTArg>>(1, padded_img_h * padded_img_w, true);
        DenseMatrix<VTArg> *selected_data = DataObjectFactory::create<DenseMatrix<VTArg>>(1, HW, true);

        if (res == nullptr)
        {
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(batch_size, CPQ, true);
        }
/*         for (uint32_t i = start; i < stop; i++)
        {
            for (uint32_t c_new = 0, off_o = oi + (i - start) * CPQ; c_new < C_new; c_new++, off_f += f_CHW)
            {
                for (uint32_t c = 0, off_i = ii + (i - start) * CHW; c < C; c++, off_i += HW)
                {
                    GetPaddedData<VTArg>(data->getValues(), padded_data->getValues(), selected_data->getValues(),
                                              pad_w, pad_h, img_w, img_h,
                                              padded_img_w, off_i);
                    for (uint32_t p = 0, off_oo = off_o; p < P; p++, off_oo += Q)
                        for (uint32_t h = p * stride_h, k = 0; h < std::min(p * stride_h + filter_h, padded_img_h); h++, k++)
                            for (uint32_t q = 0, off2 = h * padded_img_w; q < Q; q++)
                                res->getValues()[off_oo + q] = ConvOp<VTArg>(
                                    res->getValues()[off_oo + q], padded_data->getValues(), filter->getValues(),
                                    off2 + q * stride_w, off_f + k * filter_w,
                                    std::min(filter_w, padded_img_w - q * stride_w));
                }
                for (u_int32_t l = 0; l < PQ; l++)
                    res->getValues()[off_o - PQ + l] += bias->getValues()[c_new];
            }            
        } */

        u_int32_t p, h, q, w, off_o, off_i_padded, off_i = 0;
        for (uint32_t i = start; i < stop; i++)
        {            
            for (uint32_t c_new = 0; c_new < C_new; c_new++)
            {
                for (uint32_t c = 0; c < C; c++)
                {
                    off_i = ii + (i - start) * CHW + c * HW;
                    GetPaddedData<VTArg>(data->getValues(), padded_data->getValues(), selected_data->getValues(),
                                              pad_w, pad_h, img_w, img_h,
                                              padded_img_w, off_i);

                    for (p = 0; p < P; p++)
                        for (h = 0; h < std::min(filter_h, padded_img_h - p * stride_h); h++)
                            for (q = 0; q < Q; q++)
                                for(w = 0; w < std::min(filter_w, padded_img_w - q * stride_w); w++)
                                {
                                    off_o = oi + (i - start) * CPQ + c_new * PQ + p * Q + q;
                                    off_i_padded = (p * stride_h + h) * padded_img_w + q * stride_w + w;
                                    off_f = c_new * f_CHW + c * filter_h * filter_w + h * filter_w + w;

                                    // std::cout<<padded_data->getValues()[off_i_padded]<<" "<<filter->getValues()[off_f]<<std::endl;
                                    // std::cout<<filter->getValues()[off_f]<<std::endl;

                                    res->getValues()[off_o] = res->getValues()[off_o] 
                                                            + padded_data->getValues()[off_i_padded]
                                                            * filter->getValues()[off_f];
                                }
                }
                for (u_int32_t l = 0; l < PQ; l++)
                    res->getValues()[oi + (i - start) * CPQ + c_new * PQ + l] += bias->getValues()[c_new];
            }            
        }
        DataObjectFactory::destroy(padded_data);
        DataObjectFactory::destroy(selected_data);    
    }
};