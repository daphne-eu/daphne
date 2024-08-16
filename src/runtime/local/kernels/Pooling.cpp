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

#include "Pooling.h"

namespace NN::Pooling {

    uint32_t getPQ(uint32_t img_extent, uint32_t filter_extent, uint32_t pad_extent, uint32_t stride_extent) {
        uint32_t padded_image_extent = img_extent + 2 * pad_extent;
        return (padded_image_extent - filter_extent) / stride_extent + 1;
    }

    template<template<typename> class OP, typename DTRes, typename DTArg>
    void Forward<OP, DTRes, DTArg>::apply(DTRes *&res, size_t& res_h, size_t& res_w,
            const DTArg *data, const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w,
            const size_t pool_h, const size_t pool_w, const size_t stride_h, const size_t stride_w, const size_t pad_h,
            const size_t pad_w, DCTX(dctx))
    {
        auto HW = img_h * img_w;
        auto C = num_channels;
        auto CHW = C * HW;
        // padded height/width
        auto P = getPQ(img_h, pool_h, pad_h, stride_w);
        auto Q = getPQ(img_w, pool_w, pad_w, stride_h);
        auto CPQ = C * P * Q;
        res_h = P;
        res_w = Q;
        auto start = 0;
        auto stop = batch_size;

        auto ii = start * CHW;
        auto oi = start * CPQ;

        // 1 / pool length for averaging
        auto plen = static_cast<typename DTRes::VT>(1) / static_cast<typename DTRes::VT>(pool_w * pool_h);

        if (res == nullptr) {
            res = DataObjectFactory::create<DTRes>(batch_size, CPQ, true);
        }
        if (P == 1 && Q == 1 && img_w == 1) {
            //quick-path w/o materialized index arrays and
            //simplified inner loops for P = 1, Q = 1, W = 1
            uint32_t lenh = std::min(pool_h, img_h);
            for (uint32_t i = start; i < stop; i++, oi += C)
                for (uint32_t c = 0, off = ii + (i - start) * CHW; c < C; c++, off += img_h) {
                    res->getValues()[oi + c] = OP<typename DTArg::VT>::run(OP<typename DTArg::VT>::getNeutralElement(),
                                                                           data->getValues(), off, lenh, plen);
                }
        }
        else if (stride_w == 1 && stride_h == 1 && pad_h == 0 && pad_w == 0) {
            //quick-path w/o materialized index arrays
            for (uint32_t i = start; i < stop; i++)
                for (uint32_t c = 0, off = ii + (i - start) * CHW, oix = oi + (i - start) * CPQ; c < C; c++, off += HW)
                    for (uint32_t p = 0; p < P; p++, oix += Q)
                        for (uint32_t h = p; h < std::min(p + pool_h, img_h); h++)
                            for (uint32_t q = 0, off2 = off + h * img_w; q < Q; q++) {
                                res->getValues()[oix + q] = OP<typename DTArg::VT>::run(res->getValues()[oix + q],
                                        data->getValues(), off2 + q, std::min(pool_w, img_w - q), plen);
                            }
        }
        else
            throw std::runtime_error("ToDo: pooling general case with stride & padding");
    }

    template struct Forward<AVG, DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<AVG, DenseMatrix<double>, DenseMatrix<double>>;

    template struct Forward<MAX, DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<MAX, DenseMatrix<double>, DenseMatrix<double>>;
}

