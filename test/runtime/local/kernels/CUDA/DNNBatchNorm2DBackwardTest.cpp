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

#ifdef USE_CUDA
#include "run_tests.h"

#include <api/cli/DaphneUserConfig.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include "runtime/local/kernels/CUDA/BatchNorm.h"
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CheckEqApprox.h>

#include <cassert>
#include <catch.hpp>
#include <tags.h>

template<class DT>
void checkBatchNorm2DBackwardCUDA(const DT* in, const DT* dOut, const DT* gamma, const DT* mean, const DT* invVar, const DT* exp1, 
const DT* exp2, const DT* exp3, DaphneContext* dctx) {
    DT* dX = nullptr;
    DT* dGamma = nullptr;
    DT* dBeta = nullptr;

    typename DT::VT epsilon = 1e-5;
    CUDA::BatchNorm::Backward<DT, DT>::apply(dX, dGamma, dBeta, mean, invVar, in, dOut, gamma, epsilon, dctx);
    CHECK(checkEqApprox(dX, exp1, 1e-5, nullptr));
    CHECK(checkEqApprox(dGamma, exp2, 1e-4, nullptr));
    CHECK(checkEqApprox(dBeta, exp3, 1e-5, nullptr));
}

TEMPLATE_PRODUCT_TEST_CASE("batch_norm_bwd_cuda", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();
    using DT = TestType;

    auto in = genGivenVals<DT>(2, { 1, 2, 3, 4, 
                                       5, 6, 7, 8, 
                                       9, 10, 11, 12,

                                       1, 2, 3, 4, 
                                       5, 6, 7, 8, 
                                       9, 10, 11, 12});

    auto dOut = genGivenVals<DT>(2, { 1, 2, 3, 4, 
                                       5, 6, 7, 8, 
                                       9, 10, 11, 12,

                                       1, 2, 3, 4, 
                                       5, 6, 7, 8, 
                                       9, 10, 11, 12});
                                      
    auto gamma = genGivenVals<DT>(3, { 1, 1, 1 });
    auto mean = genGivenVals<DT>(3, { 2.5, 6.5, 10.5 });
    auto invVar = genGivenVals<DT>(3, { 1 / std::sqrt(1.25 + 1e-5), 1 / std::sqrt(1.25 + 1e-5), 1 / std::sqrt(1.25 + 1e-5) });
    auto res1 = genGivenVals<DT>(2, {-1.0733e-05, -3.5777e-06, 3.5777e-06,  1.0733e-05,
                                     -1.0733e-05, -3.5777e-06, 3.5777e-06,  1.0733e-05, 
                                     -1.0733e-05, -3.5777e-06, 3.5777e-06,  1.0733e-05,
                                     -1.0733e-05, -3.5777e-06, 3.5777e-06,  1.0733e-05,
                                     -1.0733e-05, -3.5777e-06, 3.5777e-06,  1.0733e-05, 
                                     -1.0733e-05, -3.5777e-06, 3.5777e-06,  1.0733e-05});
    auto res2 = genGivenVals<DT>(3, {8.9442, 8.9442, 8.9442 });
    auto res3 = genGivenVals<DT>(3, {20, 52, 84 });

    checkBatchNorm2DBackwardCUDA(in, dOut, gamma, mean, invVar, res1, res2, res3, dctx.get());
    //std::cout<<"gpu"<<std::endl;

    DataObjectFactory::destroy(in);
    DataObjectFactory::destroy(dOut);
    DataObjectFactory::destroy(gamma);
    DataObjectFactory::destroy(mean);
    DataObjectFactory::destroy(invVar);
    DataObjectFactory::destroy(res1);
    DataObjectFactory::destroy(res2);
    DataObjectFactory::destroy(res3);
}
#endif // USE_CUDA
