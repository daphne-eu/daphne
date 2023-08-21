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
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/MatMul.h>
#include <runtime/local/kernels/Transpose.h>
#include <runtime/local/kernels/Solve.h>
#include "run_tests.h"

#include <tags.h>

#include <catch.hpp>

#include <vector>

template<class DT>
void checkSolve(const DT* lhs, const DT* rhs, const DT * exp, DCTX(dctx)) {
    DT *res = nullptr;
    solve<DT, DT, DT>(res, lhs, rhs, dctx);
    // instead of CHECK(*res == * exp), we use the below approximate comparison
    // because otherwise the float results do not exactly match, while double does
    CHECK(res->getNumRows() == exp->getNumRows());
    CHECK(res->getNumCols() == exp->getNumCols());
    CHECK(checkEqApprox(res, exp, 1e-6, nullptr));
}

TEMPLATE_PRODUCT_TEST_CASE("Solve", TAG_KERNELS, (DenseMatrix), (float, double)) {
    using DT = TestType;
    auto dctx = setupContextAndLogger();

    auto X = genGivenVals<DT>(7, {
        1, 4, 5,
        3, 7, 1,
        2, 3, 5,
        9, 8, 1,
        1, 2, 3,
        5, 1, 9,
        2, 3, 1
    });
    auto w = genGivenVals<DT>(3, {
        1,
        2,
        3,
    });

    DT *y = nullptr, *tX = nullptr, *A = nullptr, *b = nullptr;
    matMul(y, X, w, false, false, dctx.get());
    transpose<DT, DT>(tX, X, dctx.get());
    matMul(A, tX, X, false, false, dctx.get());
    matMul(b, tX, y, false, false, dctx.get());

    // check solve A x = b for x
    checkSolve(A, b, w, dctx.get());

    DataObjectFactory::destroy(X);
    DataObjectFactory::destroy(w);
    DataObjectFactory::destroy(y);
    DataObjectFactory::destroy(tX);
    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(b);
}
