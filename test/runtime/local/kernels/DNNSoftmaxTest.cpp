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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>


#include <cassert>
#include <catch.hpp>
#include <tags.h>

#ifdef USE_CUDA
    #include <api/cli/DaphneUserConfig.h>
    #include "runtime/local/kernels/CUDA/Softmax.h"
    #include "runtime/local/kernels/CUDA/CreateCUDAContext.h"

template<class DT>
void check(const DT* in, const DT* exp, DaphneContext* dctx) {
    DT* res = nullptr;
    CUDA::Softmax::Forward<DT, DT>::apply(res, in, dctx);
    CHECK(Approx(*(res->getValues())).epsilon(1e-6) == *(exp->getValues()));
}

TEMPLATE_PRODUCT_TEST_CASE("softmax_fwd", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;

    DaphneUserConfig user_config{};
    auto dctx = std::make_unique<DaphneContext>(user_config);
    CUDA::createCUDAContext(dctx.get());

    auto input = genGivenVals<DT>(1, { -3, -2, -1, 0, 1, 2, 3, 4, 5});

    // expected output when used with settings filter 2x2, stride 1x1, padding 0x0
    auto result = genGivenVals<DT>(1, { 0.000212079, 0.00057649, 0.00156706, 0.00425972, 0.0115791, 0.0314753, 0.0855588,
            0.232573, 0.632199});

    check(input, result, dctx.get());

    DataObjectFactory::destroy(input);
    DataObjectFactory::destroy(result);
}

#endif // USE_CUDA