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

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include "runtime/local/kernels/CUDA/CreateCUDAContext.h"
#include <runtime/local/kernels/Fill.h>
#include <runtime/local/kernels/CUDA/Fill.h>

#include <tags.h>
#include <catch.hpp>

template<class DTRes>
void checkFill(const typename DTRes::VT val, const size_t rows, const size_t cols, DaphneContext* ctx) {
    DTRes* res = nullptr;
    DTRes* exp = nullptr;
    CUDA::Fill<DTRes, typename DTRes::VT>::apply(res, val, rows, cols, ctx);
    fill<DTRes, typename DTRes::VT>(exp, val, rows, cols, ctx);

    CHECK(*res == *exp);

    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(exp);
}

TEMPLATE_PRODUCT_TEST_CASE("CUDA::fill", TAG_KERNELS, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();

    checkFill<TestType>(0, 123, 456, dctx.get());
    checkFill<TestType>(123.45, 123, 456, dctx.get());
}

TEMPLATE_PRODUCT_TEST_CASE("CUDA::fill", TAG_KERNELS, (DenseMatrix), (int64_t, uint8_t)) { // NOLINT(cert-err58-cpp)
    auto dctx = setupContextAndLogger();

    checkFill<TestType>(0, 123, 456, dctx.get());
    checkFill<TestType>(123, 123, 456, dctx.get());
}
