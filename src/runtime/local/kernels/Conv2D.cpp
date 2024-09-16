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

#include "Conv2DForward.h"

namespace NN::Conv2D {

    uint32_t getPQ(uint32_t img_extent, uint32_t filter_extent, uint32_t pad_extent, uint32_t stride_extent) {
        uint32_t padded_image_extent = img_extent + 2 * pad_extent;
        return (padded_image_extent - filter_extent) / stride_extent + 1;
    }

    template<typename DTRes, typename DTArg>
    void Forward<DTRes, DTArg>::apply(DTRes *&res, size_t& res_h, size_t& res_w,
                          const DTArg *data, const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w,
                          const DTArg *filter, const size_t num_filters, const size_t filter_h, const size_t filter_w, 
                          const size_t stride_h, const size_t stride_w, 
                          const size_t pad_h, const size_t pad_w, DCTX(dctx))
    {
        auto HW = img_h * img_w;
        auto C = num_channels;
        auto CHW = C * HW;
        // padded height/width
        auto P = getPQ(img_h, filter_h, pad_h, stride_w);
        auto Q = getPQ(img_w, filter_w, pad_w, stride_h);
        auto CPQ = C * P * Q;
        res_h = P;
        res_w = Q;
        auto start = 0;
        auto stop = batch_size;

        auto ii = start * CHW;
        auto oi = start * CPQ;

        auto padded_img_h = img_h + 2 * pad_h;
        auto padded_img_w = img_w + 2 * pad_w;
        DTArg *padded_data = DataObjectFactory::create<DTArg>(1, padded_img_h * padded_img_w, true);
        DTArg *selected_data = DataObjectFactory::create<DTArg>(1, HW, true);
        
        if (res == nullptr) {
            res = DataObjectFactory::create<DTRes>(batch_size, CPQ, true);
        }        
        for (uint32_t i = start; i < stop; i++)
            for (uint32_t c = 0, off = ii + (i - start) * CHW, oix = oi + (i - start) * CPQ; c < C; c++, off += HW){                  
                GetPaddedData<typename DTArg::VT>::run(data->getValues(),
                                                           padded_data->getValues(),
                                                           selected_data->getValues(),
                                                           pad_w, pad_h, img_w, img_h,
                                                           padded_img_w, off);
            for (uint32_t p = 0; p < P; p++, oix += Q)
                for (uint32_t h = p * stride_h; h < std::min(p * stride_h + pool_h, padded_img_h); h++)
                    for (uint32_t q = 0, off2 = h * padded_img_w; q < Q; q++) 
                        res->getValues()[oix + q] = Conv2D<typename DTArg::VT>::run(
                            res->getValues()[oix + q], padded_data->getValues(), off2 + q * stride_w, std::min(pool_w, padded_img_w - q * stride_w), plen);
                            
                }
                    
        

    }

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;
}

