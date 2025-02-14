/*
 * Copyright 2025 The DAPHNE Consortium
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

#include "runtime/local/datagen/GenGivenVals.h"
#include "runtime/local/kernels/CUDA/BiasAdd.h"

template <class DT>
void checkBiasAdd(const DT* data, const DT* bias, const DT* expected, DaphneContext* dctx)
{
    DT* res = nullptr;
    CUDA::BiasAdd<DT, DT>::apply(res, data, bias, dctx);

    CHECK(Approx(*(res->getValues())).epsilon(1e-6) == *(expected->getValues()));

    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("CUDA::BiasAdd", TAG_KERNELS, (DenseMatrix), (float, double)) {
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto data = genGivenVals<DT>(/*numRows=*/1, {0.0, 1.0, 2.0});
    auto bias = genGivenVals<DT>(/*numRows=*/1, {1.0, 2.0, 3.0});
    auto expected = genGivenVals<DT>(/*numRows=*/1, {1.0, 3.0, 5.0});

    checkBiasAdd<DT>(data, bias, expected, dctx.get());

    // Clean up
    DataObjectFactory::destroy(data);
    DataObjectFactory::destroy(bias);
    DataObjectFactory::destroy(expected);
}

