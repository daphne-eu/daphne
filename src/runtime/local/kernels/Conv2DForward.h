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

#include "Padding.h"

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct Conv2DForward {
    static void apply(DTRes *&res, size_t &res_h, size_t &res_w,
                      const DTArg *data, const DTArg *filter, const DTArg *bias,
                      const size_t batch_size, const size_t num_channels,
                      const size_t img_h, const size_t img_w,
                      const size_t filter_h, const size_t filter_w,
                      const size_t stride_h, const size_t stride_w,
                      const size_t pad_h, const size_t pad_w,
                      DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void conv2DForward(DTRes *&res, size_t &res_h, size_t &res_w, const DTArg *data,
                   const DTArg *filter, const DTArg *bias,
                   const size_t batch_size, const size_t num_channels,
                   const size_t img_h, const size_t img_w,
                   const size_t filter_h, const size_t filter_w,
                   const size_t stride_h, const size_t stride_w,
                   const size_t pad_h, const size_t pad_w, DCTX(dctx)) {
    Conv2DForward<DTRes, DTArg>::apply(
        res, res_h, res_w, data, filter, bias, batch_size, num_channels, img_h,
        img_w, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
static inline VT ConvOp(VT initial_value, const VT *in, const VT *filter,
                        uint32_t in_start, uint32_t filter_start,
                        uint32_t length) {
    VT ret = 0;
    for (uint32_t i = 0; i < length; ++i)
        ret += in[in_start + i] * filter[filter_start + i];
    return ret + initial_value;
}

template <typename VTRes, typename VTArg>
struct Conv2DForward<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, size_t &res_h, size_t &res_w,
                      const DenseMatrix<VTArg> *data,
                      const DenseMatrix<VTArg> *filter,
                      const DenseMatrix<VTArg> *bias, const size_t batch_size,
                      const size_t num_channels, const size_t img_h,
                      const size_t img_w, const size_t filter_h,
                      const size_t filter_w, const size_t stride_h,
                      const size_t stride_w, const size_t pad_h,
                      const size_t pad_w, DCTX(dctx)) {
        auto HW = img_h * img_w;
        auto C = num_channels;
        auto CHW = C * HW;
        // padded height/width
        auto P = getPQ(img_h, filter_h, pad_h, stride_h);
        auto Q = getPQ(img_w, filter_w, pad_w, stride_w);
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
        DenseMatrix<VTArg> *padded_data =
            DataObjectFactory::create<DenseMatrix<VTArg>>(
                1, padded_img_h * padded_img_w, true);

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VTArg>>(batch_size, CPQ,
                                                                true);
        }

        u_int32_t off_o, off_i_padded, off_i = 0;
        for (uint32_t i = start; i < stop; i++) {
            for (uint32_t c_new = 0; c_new < C_new; c_new++) {
                for (uint32_t c = 0; c < C; c++) {
                    off_i = ii + (i - start) * CHW + c * HW;

                    Padding(padded_data->getValues(), data->getValues(), pad_h,
                            pad_w, img_w, img_h, off_i);

                    for (u_int32_t p = 0; p < P; p++)
                        for (u_int32_t h = 0;
                             h <
                             std::min(filter_h, padded_img_h - p * stride_h);
                             h++)
                            for (u_int32_t q = 0; q < Q; q++)
                                for (u_int32_t w = 0;
                                     w < std::min(filter_w,
                                                  padded_img_w - q * stride_w);
                                     w++) {
                                    off_o = oi + (i - start) * CPQ +
                                            c_new * PQ + p * Q + q;
                                    off_i_padded =
                                        (p * stride_h + h) * padded_img_w +
                                        q * stride_w + w;
                                    off_f = c_new * f_CHW +
                                            c * filter_h * filter_w +
                                            h * filter_w + w;

                                    res->getValues()[off_o] =
                                        res->getValues()[off_o] +
                                        padded_data->getValues()[off_i_padded] *
                                            filter->getValues()[off_f];
                                }
                }
                for (u_int32_t l = 0; l < PQ; l++)
                    res->getValues()[oi + (i - start) * CPQ + c_new * PQ + l] +=
                        bias->getValues()[c_new];
            }
        }
        DataObjectFactory::destroy(padded_data);
    }
};