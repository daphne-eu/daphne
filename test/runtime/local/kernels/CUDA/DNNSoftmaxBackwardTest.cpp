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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEqApprox.h>

#include <catch.hpp>
#include <tags.h>

#ifdef USE_CUDA
    #include "runtime/local/kernels/CUDA/Softmax.h"

template<class DT>
void check(const DT* output, const DT* dOutput, const DT* exp, DaphneContext* dctx) {
    DT* res = nullptr;
    CUDA::Softmax::Backward<DT, DT>::apply(res, output, dOutput, dctx);
    //CHECK(Approx(*(res->getValues())).epsilon(1e-6) == *(exp->getValues()));
    CHECK(checkEqApprox(res, exp, 1e-5, nullptr));
}

TEMPLATE_PRODUCT_TEST_CASE("softmax_bwd_cuda", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto result = genGivenVals<DT>(
        1, {-0.00157, -0.00370, -0.00849, -0.01882, -0.03959, -0.07614, -0.12142, -0.09748, 0.36722});
        
    auto output = genGivenVals<DT>(
        1, {0.000212079, 0.00057649, 0.00156706, 0.00425972, 0.0115791, 0.0314753, 0.0855588, 0.232573, 0.632199});
           
    auto dOutput = genGivenVals<DT>(1, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    check(output, dOutput, result, dctx.get());

    DataObjectFactory::destroy(output);
    DataObjectFactory::destroy(dOutput);
    DataObjectFactory::destroy(result);
}

#endif // USE_CUDA