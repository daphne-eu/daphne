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

#ifdef USE_CUDA

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>

#include <cassert>
#include <tags.h>
#include <catch.hpp>

#include "runtime/local/kernels/CUDA/Convolution.h"

template<class DT>
void check(const DT* in, const DT* filter, const DT* exp, DaphneContext* dctx) {
    DT* res = nullptr;
    size_t out_h;
    size_t out_w;
    CUDA::Convolution::Forward<DT, DT>::apply(res, out_h, out_w, in, filter, nullptr, in->getNumRows(), 1, 3, 3, 2, 2,
            1, 1, 0, 0, dctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("conv_fwd", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto input = genGivenVals<DT>(1, { 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto filter = genGivenVals<DT>(1, { 1, 0, 0, 1});

    // expected output when used with settings filter 2x2, stride 1x1, padding 0x0
    auto result = genGivenVals<DT>(1, { 6, 8, 12, 14 });

    check(input, filter, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}

#endif // USE_CUDA