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
#include <runtime/local/kernels/DiagVector.h>


#include <tags.h>

#include <catch.hpp>

#include <type_traits>

#include <cstdint>

#define DATA_TYPES CSRMatrix, DenseMatrix, Matrix
#define VALUE_TYPES  int32_t, double  

template<class DTRes, class DTArg>
void checkDiagVector(const DTArg * arg, DTRes *& res, DTRes * exp) {
    diagVector<DTRes, DTArg>(res, arg, nullptr);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("DiagVector-normal", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){ // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using VT = typename DT::VT;
    using DTRes = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        Matrix<VT>,
                        DenseMatrix<VT>
                    >::type;

    DT * arg = genGivenVals<DT>(3, {
        3,0,0,
        0,2,0,
        0,0,1,
    });
    DTRes * exp = genGivenVals<DTRes>(3, {3, 2, 1});
    DTRes * res = nullptr;

    checkDiagVector(arg, res, exp);

    DataObjectFactory::destroy(arg, exp, res);
}


TEMPLATE_PRODUCT_TEST_CASE("DiagVector-mixed-diagonal", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){ // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using VT = typename DT::VT;
    using DTRes = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        Matrix<VT>,
                        DenseMatrix<VT>
                    >::type;

    DT * arg = genGivenVals<DT>(3, {
        1,0,0,
        0,0,0,
        0,0,1,
    });
    DTRes * exp = genGivenVals<DTRes>(3, {1, 0, 1});
    DTRes * res = nullptr;

    checkDiagVector(arg, res, exp);

    DataObjectFactory::destroy(arg, exp, res);
}


TEMPLATE_PRODUCT_TEST_CASE("DiagVector-null", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){ // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using VT = typename DT::VT;
    using DTRes = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        Matrix<VT>,
                        DenseMatrix<VT>
                    >::type;

    DT * arg = genGivenVals<DT>(3, {
        3,0,0,
        0,2,0,
        0,0,1,
    });
    DTRes * exp = genGivenVals<DTRes>(3, {3, 2, 1});
    DTRes * res = nullptr;
    
    checkDiagVector(arg, res, exp);

    DataObjectFactory::destroy(arg, exp, res);
}
