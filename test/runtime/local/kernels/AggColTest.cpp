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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/AggCol.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TEST_NAME(opName) "AggCol (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix
#define VALUE_TYPES double, uint32_t

template<class DTRes, class DTArg>
void checkAggCol(AggOpCode opCode, const DTArg * arg, const DTRes * exp) {
    DTRes * res = nullptr;
    aggCol<DTRes, DTArg>(opCode, res, arg, nullptr);
    if constexpr(std::is_same_v<DTRes, DenseMatrix<StringScalarType>>)
        for(size_t val = 0; val < exp->getNumItems(); val++)
            CHECK(strcmp(res->getValues()[val], exp->getValues()[val]) == 0);
    else
        CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<typename DTArg::VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        3, 0, 2, 0,
        0, 0, 1, 1,
        2, 5, 0, 0,
    });
    auto m1exp = genGivenVals<DTRes>(1, {5, 5, 3, 1});
    
    checkAggCol(AggOpCode::SUM, m0, m0exp);
    checkAggCol(AggOpCode::SUM, m1, m1exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<typename DTArg::VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        4, 6, 3, 9,
        5, 2, 0, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DTRes>(1, {4, 2, 0, 4});
    auto m2 = genGivenVals<DTArg>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    auto m2exp = genGivenVals<DTRes>(1, {0, 0, 0, 0});
    auto m3 = genGivenVals<DenseMatrix<const char*>>(3, {
        "Zambia", "Australia", "UAE", "India",
        "Portugal", "Germany", "Zimbabwe", "Vatican",
        "Austria", "Finland", "Canada", "Iceland",
    });
    auto m3exp = genGivenVals<DenseMatrix<const char*>>(1, {"Austria", "Australia", "Canada", "Iceland"});
    
    checkAggCol(AggOpCode::MIN, m0, m0exp);
    checkAggCol(AggOpCode::MIN, m1, m1exp);
    checkAggCol(AggOpCode::MIN, m2, m2exp);
    checkAggCol(AggOpCode::MIN, m3, m3exp);

    DataObjectFactory::destroy(m0, m0exp, m1, m1exp, m2, m2exp, m3, m3exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<typename DTArg::VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        4, 6, 3, 9,
        5, 2, 0, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DTRes>(1, {7, 6, 5, 9});
    auto m2 = genGivenVals<DTArg>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    auto m2exp = genGivenVals<DTRes>(1, {4, 2, 5, 9});

    auto m3 = genGivenVals<DenseMatrix<const char*>>(3, {
        "Zambia", "Australia", "UAE", "India",
        "Portugal", "Germany", "Zimbabwe", "Vatican",
        "Austria", "Finland", "Canada", "Iceland",
    });
    auto m3exp = genGivenVals<DenseMatrix<const char*>>(1, {"Zambia", "Germany", "Zimbabwe", "Vatican"});
    
    checkAggCol(AggOpCode::MAX, m0, m0exp);
    checkAggCol(AggOpCode::MAX, m1, m1exp);
    checkAggCol(AggOpCode::MAX, m2, m2exp);
    checkAggCol(AggOpCode::MAX, m3, m3exp);
    
DataObjectFactory::destroy(m0, m0exp, m1, m1exp, m2, m2exp, m3, m3exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean"), TAG_KERNELS, (DATA_TYPES), (int64_t, double)) {
    using DTArg = TestType;
    using VT = typename DTArg::VT;
    using DTRes = DenseMatrix<VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0});
    auto m2 = genGivenVals<DTArg>(4, {
        1, 3, 0, -1,
        1, 3, 5,  3,
        3, 1, 0,  0,
        3, 1, 5, -1,
    });
    auto m2exp = genGivenVals<DTRes>(1, {2, 2, VT(2.5), VT(0.25)});
    
    checkAggCol(AggOpCode::MEAN, m0, m0exp);
    checkAggCol(AggOpCode::MEAN, m2, m2exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m2exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("stddev"), TAG_KERNELS, (DATA_TYPES), (int64_t, double)) {
    using DTArg = TestType;
    using VT = typename DTArg::VT;
    using DTRes = DenseMatrix<VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0});
    auto m2 = genGivenVals<DTArg>(4, {
        1, 3, 0, -1,
        1, 3, 5,  3,
        3, 1, 0,  0,
        3, 1, 5, -1,
    });
    auto m2exp = genGivenVals<DTRes>(1, {1, 1, VT(2.5), VT(1.6393596310755)});
    
    checkAggCol(AggOpCode::STDDEV, m0, m0exp);
    checkAggCol(AggOpCode::STDDEV, m2, m2exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m2exp);
}
