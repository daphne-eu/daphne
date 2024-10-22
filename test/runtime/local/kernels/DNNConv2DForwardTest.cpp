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

#include "run_tests.h"

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/Conv2DForward.h>

template <class DT> void checkConv2DForward(const DT *in, const DT *filter, const DT *exp, DaphneContext *dctx) {
    DT *res = nullptr;
    size_t out_h;
    size_t out_w;
    auto bias = genGivenVals<DT>(1, {0});
    Conv2DForward<DT, DT>::apply(res, out_h, out_w, in, filter, bias, in->getNumRows(), 3, 3, 3, 2, 2, 2, 2, 1, 1,
                                 dctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("conv_fwd_cpu", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto input = genGivenVals<DT>(2, {
                                         1,  2,  3,  4,  5,  6,  7,  8,  9,

                                         10, 11, 12, 13, 14, 15, 16, 17, 18,

                                         19, 20, 21, 22, 23, 24, 25, 26, 27,

                                         1,  2,  3,  4,  5,  6,  7,  8,  9,

                                         10, 11, 12, 13, 14, 15, 16, 17, 18,

                                         19, 20, 21, 22, 23, 24, 25, 26, 27,
                                     });

    auto filter = genGivenVals<DT>(1, {1, 0, 0, 1,

                                       1, 0, 0, 1,

                                       1, 0, 0, 1});

    // expected output when used with settings filter 2x2, stride 1x1, padding
    // 0x0 auto result = genGivenVals<DT>(1, { 6, 8, 12, 14 });
    auto result = genGivenVals<DT>(2, {30, 36, 48, 96, 30, 36, 48, 96});

    checkConv2DForward(input, filter, result, dctx.get());

    DataObjectFactory::destroy(filter);
    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}
