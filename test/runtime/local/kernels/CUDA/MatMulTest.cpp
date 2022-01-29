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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CUDA_InitContext.h>
#include <runtime/local/kernels/CUDA/MatMul.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

template<class DT>
void checkMatMulCUDA(const DT * lhs, const DT * rhs, const DT * exp, DaphneContext* ctx) {
    DT* res = nullptr;
    CUDA::matMul<DT, DT, DT>(res, lhs, rhs, ctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("CUDA::matMul", TAG_KERNELS, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;

    DaphneUserConfig user_config{};
    auto dctx = std::make_unique<DaphneContext>(user_config);
    initCUDAContext(dctx.get());

    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        1, 2, 3,
        3, 1, 2,
        2, 3, 1,
    });
    auto m2 = genGivenVals<DT>(3, {
        13, 13, 10,
        10, 13, 13,
        13, 10, 13,
    });
    auto m3 = genGivenVals<DT>(2, {
        1, 0, 3, 0,
        0, 0, 2, 0,
    });
    auto m4 = genGivenVals<DT>(4, {
        0, 1,
        2, 0,
        1, 1,
        0, 0,
    });
    auto m5 = genGivenVals<DT>(2, {
        3, 4,
        2, 2,
    });
    auto v0 = genGivenVals<DT>(3, {
        0,
        0,
        0
    });
    auto v1 = genGivenVals<DT>(3, {
        1,
        1,
        1
    });
    auto v2 = genGivenVals<DT>(3, {
        1,
        2,
        3
    });
    auto v3 = genGivenVals<DT>(3, {
        6,
        6,
        6
    });
    auto v4 = genGivenVals<DT>(3, {
        14,
        11,
        11
    });

    checkMatMulCUDA(m0, m0, m0, dctx.get());
    checkMatMulCUDA(m1, m1, m2, dctx.get());
    checkMatMulCUDA(m3, m4, m5, dctx.get());
    checkMatMulCUDA(m0, v0, v0, dctx.get());
    checkMatMulCUDA(m1, v0, v0, dctx.get());
    checkMatMulCUDA(m2, v0, v0, dctx.get());
    checkMatMulCUDA(m0, v1, v0, dctx.get());
    checkMatMulCUDA(m1, v1, v3, dctx.get());
    checkMatMulCUDA(m1, v2, v4, dctx.get());

    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
    DataObjectFactory::destroy(m4);
    DataObjectFactory::destroy(m5);
    DataObjectFactory::destroy(v0);
    DataObjectFactory::destroy(v1);
    DataObjectFactory::destroy(v2);
    DataObjectFactory::destroy(v3);
    DataObjectFactory::destroy(v4);
}