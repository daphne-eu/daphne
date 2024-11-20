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

#include "run_tests.h"

#ifdef USE_CUDA


#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>

#include <cassert>
#include <tags.h>
#include <catch.hpp>

#include "runtime/local/kernels/CUDA/Convolution.h"

template<class DT>
void checkConv2DBackwardData(
    const DT* filter, const DT* dOutput, const DT* exp, DaphneContext* dctx) 
{
    DT* res = nullptr;
    CUDA::Convolution::Backward<DT, DT>::Data::apply(
        res, filter, dOutput, 
        1, 3, //batch_size, num_channels
        3, 3, //img_h, img_w
        2, 2, //filter_h, filter_w
        2, 2, //stride_h, stride_w
        1, 1, //pad_h, pad_w     
        dctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("conv_bwd_data_cuda", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto dOutput = genGivenVals<DT>(1, { 1, 2, 3, 4, 
                                       5, 6, 7, 8});

    auto filter = genGivenVals<DT>(2, { 1, 0, 0, 2, 
                                        2, 0, 0, 3, 
                                        3, 0, 0, 4, 
                                        
                                        5, 0, 0, 6, 
                                        6, 0, 0, 7,
                                        7, 0, 0, 8});

    auto result = genGivenVals<DT>(1, { 32, 0, 40, 0, 44, 0, 48, 0, 56,
                                        38, 0, 48, 0, 56, 0, 58, 0, 68,
                                        44, 0, 56, 0, 68, 0, 68, 0, 80 });

    checkConv2DBackwardData(filter, dOutput, result, dctx.get());

    DataObjectFactory::destroy(filter);
    DataObjectFactory::destroy(dOutput);
    DataObjectFactory::destroy(result);
}

#endif // USE_CUDA