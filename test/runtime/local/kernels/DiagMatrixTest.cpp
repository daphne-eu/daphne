/*
 * copyright 2021 The DAPHNE Consortium
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

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/DiagMatrix.h>

#include <tags.h>
#include <cstdint>
#include <catch.hpp>

#define TEST_NAME(opName) "DiagMatrix (" opName ")"
#define VALUE_TYPES int32_t, float

template<class DTRes, class DTArg>
void checkDiagMatrix(const DTArg * arg, const DTRes * exp) {
    DTRes * res = nullptr;
    diagMatrix<DTRes, DTArg>(res, arg, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("diag-dense"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        3,
        1,
        2,
    });
    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        3,0,0,
        0,1,0,
        0,0,2,
    });
    auto csr_exp = genGivenVals<CSRMatrix<VT>>(3, {
        3,0,0,
        0,1,0,
        0,0,2,
    });

    checkDiagMatrix(arg, dense_exp);
    checkDiagMatrix(arg, csr_exp);

    DataObjectFactory::destroy(csr_exp, dense_exp, arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("diag-csr/generic"), TAG_KERNELS, (CSRMatrix, Matrix), (VALUE_TYPES)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;

    auto arg = genGivenVals<DT>(5, {
        3,
        0,
        0,
        1,
        0,
    });
    auto exp = genGivenVals<DT>(5, {
        3,0,0,0,0,
        0,0,0,0,0,
        0,0,0,0,0,
        0,0,0,1,0,
        0,0,0,0,0,
    });

    checkDiagMatrix(arg, exp);

    DataObjectFactory::destroy(exp, arg);

    arg = genGivenVals<DT>(3, {
        3,
        1,
        2,
    });
    exp = genGivenVals<DT>(3, {
        3,0,0,
        0,1,0,
        0,0,2,
    });

    checkDiagMatrix(arg, exp);

    DataObjectFactory::destroy(exp, arg);
}
