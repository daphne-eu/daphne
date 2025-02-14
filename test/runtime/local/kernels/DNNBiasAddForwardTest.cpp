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

#include "runtime/local/kernels/BiasAdd.h"
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <catch.hpp>
#include <tags.h>

template <class DT> void checkBiasAddForward(const DT *input, const DT *bias, const DT *exp, DaphneContext *dctx) {
    DT *res = nullptr;
    BiasAdd<DT, DT>::apply(res, input, bias, dctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("bias_add_fwd", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto input = genGivenVals<DT>(2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,

                                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto bias = genGivenVals<DT>(3, {1, 2, 3});

    auto result = genGivenVals<DT>(2, {2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15,

                                       2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15});

    checkBiasAddForward(input, bias, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(bias);
    DataObjectFactory::destroy(result);
}
