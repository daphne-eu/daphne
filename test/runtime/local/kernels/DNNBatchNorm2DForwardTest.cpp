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

#include "runtime/local/kernels/BatchNorm2DTestForward.h"
#include "runtime/local/kernels/BatchNorm2DTrainForward.h"
#include <runtime/local/datagen/GenGivenVals.h>

template <class DT>
void checkBatchNorm2DTestForward(const DT *in, const DT *gamma, const DT *beta, const DT *ema_mean, const DT *ema_var,
                                 const DT *exp, DaphneContext *dctx) {
    DT *res = nullptr;
    typename DT::VT epsilon = 1e-5;
    BatchNorm2DTestForward<DT, DT>::apply(res, in, gamma, beta, ema_mean, ema_var, epsilon, dctx);
    CHECK(Approx(*(res->getValues())).epsilon(epsilon) == *(exp->getValues()));
}

TEMPLATE_PRODUCT_TEST_CASE("batch_norm_test_fwd", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto input = genGivenVals<DT>(1, {-3, -2, -1, 0, 1, 2, 3, 4, 5});
    auto gamma = genGivenVals<DT>(1, {1});
    auto beta = genGivenVals<DT>(1, {0});
    auto ema_mean = genGivenVals<DT>(1, {0});
    auto ema_var = genGivenVals<DT>(1, {1});

    auto result = genGivenVals<DT>(1, {-3, -2, -1, 0, 1, 2, 3, 4, 5});

    checkBatchNorm2DTestForward(input, gamma, beta, ema_mean, ema_var, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}

template <class DT>
void checkBatchNorm2DTrainForward(const DT *in, const DT *gamma, const DT *beta, const DT *ema_mean, const DT *ema_var,
                                  const DT *exp, DaphneContext *dctx) {
    DT *res = nullptr;
    DT *new_emaMean = nullptr;
    DT *new_emaVar = nullptr;
    DT *Mean = nullptr;
    DT *invVar = nullptr;
    typename DT::VT epsilon = 1e-5;
    typename DT::VT mu = 1;
    BatchNorm2DTrainForward<DT, DT>::apply(res, new_emaMean, new_emaVar, Mean, invVar, in, gamma, beta, ema_mean,
                                           ema_var, epsilon, mu, dctx);
    CHECK(Approx(*(res->getValues())).epsilon(epsilon) == *(exp->getValues()));
}

TEMPLATE_PRODUCT_TEST_CASE("batch_norm_train_fwd", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto input =
        genGivenVals<DT>(2, {-3, -2, -1, 0, 1, 2, 3, 4, 5, -3, -2, -1, 0, 1, 2, 3, 4, 5, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                             -3, -2, -1, 0, 1, 2, 3, 4, 5, -3, -2, -1, 0, 1, 2, 3, 4, 5, -3, -2, -1, 0, 1, 2, 3, 4, 5});
    auto gamma = genGivenVals<DT>(3, {1, 1, 1});
    auto beta = genGivenVals<DT>(3, {0, 0, 0});
    auto ema_mean = genGivenVals<DT>(3, {0, 0, 0});
    auto ema_var = genGivenVals<DT>(1, {1, 1, 1});

    auto result = genGivenVals<DT>(2, {-1.5492, -1.1619, -0.7746, -0.3873, 0.0000, 0.3873, 0.7746, 1.1619, 1.5492,
                                       -1.5492, -1.1619, -0.7746, -0.3873, 0.0000, 0.3873, 0.7746, 1.1619, 1.5492,
                                       -1.5492, -1.1619, -0.7746, -0.3873, 0.0000, 0.3873, 0.7746, 1.1619, 1.5492,
                                       -1.5492, -1.1619, -0.7746, -0.3873, 0.0000, 0.3873, 0.7746, 1.1619, 1.5492,
                                       -1.5492, -1.1619, -0.7746, -0.3873, 0.0000, 0.3873, 0.7746, 1.1619, 1.5492,
                                       -1.5492, -1.1619, -0.7746, -0.3873, 0.0000, 0.3873, 0.7746, 1.1619, 1.5492});

    checkBatchNorm2DTrainForward(input, gamma, beta, ema_mean, ema_var, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}