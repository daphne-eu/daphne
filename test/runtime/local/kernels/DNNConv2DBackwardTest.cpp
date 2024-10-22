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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/Conv2DBackwardData.h>
#include <runtime/local/kernels/Conv2DBackwardFilter.h>

template<class DT>
void checkConv2DBackwardData(const DT* in, const DT* filter, const DT* exp, DaphneContext* dctx) {
    DT* res = nullptr;
    Conv2DBackwardData<DT, DT>::apply(filter, in, 2, 2, 1, 1, 1, 3, 3, 3, 2, 3, 2, 2, res, dctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("conv_bwd_data", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    // auto input = genGivenVals<DT>(1, { 1, 2, 3, 4, 5, 6, 7, 8, 9});
    // auto filter = genGivenVals<DT>(1, { 1, 0, 0, 1});

    auto input = genGivenVals<DT>(1, { 1, 2, 3, 4, 
                                       5, 6, 7, 8});

    auto filter = genGivenVals<DT>(2, { 1, 0, 0, 2, 
                                        2, 0, 0, 3, 
                                        3, 0, 0, 4, 
                                        
                                        5, 0, 0, 6, 
                                        6, 0, 0, 7,
                                        7, 0, 0, 8});

    // expected output when used with settings filter 2x2, stride 1x1, padding 0x0
    // auto result = genGivenVals<DT>(1, { 6, 8, 12, 14 });
    auto result = genGivenVals<DT>(1, { 32, 0, 40, 0, 44, 0, 48, 0, 56,
                                        38, 0, 48, 0, 56, 0, 58, 0, 68,
                                        44, 0, 56, 0, 68, 0, 68, 0, 80 });

    checkConv2DBackwardData(input, filter, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}

template<class DT>
void checkConv2DBackwardFilter(const DT* input, const DT* dOutput, const DT* exp, DaphneContext* dctx) {
    DT* res = nullptr;
    Conv2DBackwardFilter<DT, DT>::apply(res, input, dOutput, 2, 2, 1, 1, 1, 3, 3, 3, 2, 3, 2, 2, dctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("conv_bwd_filter", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    // auto input = genGivenVals<DT>(1, { 1, 2, 3, 4, 5, 6, 7, 8, 9});
    // auto filter = genGivenVals<DT>(1, { 1, 0, 0, 1});

    auto input = genGivenVals<DT>(1, { 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                                       1, 2, 3, 4, 5, 6, 7, 8, 9 });

    auto dOutput = genGivenVals<DT>(1, { 1, 2, 3, 4,
                                         5, 6, 7, 8});

    auto filter = genGivenVals<DT>(2, { 1, 0, 0, 2,
                                        2, 0, 0, 3,
                                        3, 0, 0, 4,

                                        5, 0, 0, 6,
                                        6, 0, 0, 7,
                                        7, 0, 0, 8});

    // expected output when used with settings filter 2x2, stride 1x1, padding 0x0
    // auto result = genGivenVals<DT>(1, { 6, 8, 12, 14 });
    auto result = genGivenVals<DT>(2, { 20, 36, 36, 64,
                                        20, 36, 36, 64,
                                        20, 36, 36, 64,

                                        40, 76, 76, 144,
                                        40, 76, 76, 144,
                                        40, 76, 76, 144 });

    checkConv2DBackwardFilter(input, dOutput, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
    DataObjectFactory::destroy(dOutput);
    DataObjectFactory::destroy(filter);
}
