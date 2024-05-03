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

#ifdef USE_CUDA

#include "run_tests.h"

#include <api/cli/DaphneUserConfig.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include "runtime/local/kernels/CUDA/BatchNorm.h"

#include <cassert>
#include <catch.hpp>
#include <tags.h>

template<class DT>
void check(const DT* in, const DT* gamma, const DT* beta, const DT* ema_mean, const DT* ema_var, const DT* exp,
        DaphneContext* dctx)
{
    DT* res = nullptr;
    typename DT::VT epsilon = 1e-5;
    CUDA::BatchNorm::Forward<DT, DT>::apply(res, in, gamma, beta, ema_mean, ema_var, epsilon, dctx);
    CHECK(Approx(*(res->getValues())).epsilon(epsilon) == *(exp->getValues()));
}

TEMPLATE_PRODUCT_TEST_CASE("CUDA::BatchNorm::Forward", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto input = genGivenVals<DT>(1, { -3, -2, -1, 0, 1, 2, 3, 4, 5});
    auto gamma = genGivenVals<DT>(1, { 1 });
    auto beta = genGivenVals<DT>(1, { 0 });
    auto ema_mean = genGivenVals<DT>(1, { 0 });
    auto ema_var = genGivenVals<DT>(1, { 1 });

    auto result = genGivenVals<DT>(1, { -3, -2, -1, 0, 1, 2, 3, 4, 5});

    check(input, gamma, beta, ema_mean, ema_var, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}

#endif // USE_CUDA