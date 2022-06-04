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
#include <runtime/local/kernels/AggRow.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TEST_NAME(opName) "AggRow (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix
#define VALUE_TYPES double, uint32_t

template<class DTRes, class DTArg>
void checkAggRow(AggOpCode opCode, const DTArg * arg, const DTRes * exp) {
    DTRes * res = nullptr;
    aggRow<DTRes, DTArg>(opCode, res, arg, nullptr);
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
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        3, 0, 2, 0,
        0, 0, 1, 1,
        2, 5, 0, 0,
    });
    auto m1exp = genGivenVals<DTRes>(3, {5, 2, 7});
    
    checkAggRow(AggOpCode::SUM, m0, m0exp);
    checkAggRow(AggOpCode::SUM, m1, m1exp);
    
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
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        4, 6, 3, 9,
        5, 2, 0, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DTRes>(3, {3, 0, 4});
    auto m2 = genGivenVals<DTArg>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    auto m2exp = genGivenVals<DTRes>(3, {0, 0, 0});
    
    checkAggRow(AggOpCode::MIN, m0, m0exp);
    checkAggRow(AggOpCode::MIN, m1, m1exp);
    checkAggRow(AggOpCode::MIN, m2, m2exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m2exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<typename DTArg::VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        4, 6, 3, 9,
        5, 2, 0, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DTRes>(3, {9, 9, 7});
    auto m2 = genGivenVals<DTArg>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    auto m2exp = genGivenVals<DTRes>(3, {9, 2, 5});
    
    checkAggRow(AggOpCode::MAX, m0, m0exp);
    checkAggRow(AggOpCode::MAX, m1, m1exp);
    checkAggRow(AggOpCode::MAX, m2, m2exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m2exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("idxmin"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<typename DTArg::VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        4, 6, 3, 9,
        2, 5, 0, 1,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DTRes>(3, {2, 2, 1});
    auto m2 = genGivenVals<DTArg>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    auto m2exp = genGivenVals<DTRes>(3, {1, 0, 0});
    
    checkAggRow(AggOpCode::IDXMIN, m0, m0exp);
    checkAggRow(AggOpCode::IDXMIN, m1, m1exp);
    checkAggRow(AggOpCode::IDXMIN, m2, m2exp);
    
    DataObjectFactory::destroy(m0, m0exp, m1, m1exp, m2, m2exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("idxmax"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<typename DTArg::VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        4, 6, 3, 9,
        2, 5, 0, 1,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DTRes>(3, {3, 1, 0});
    auto m2 = genGivenVals<DTArg>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    auto m2exp = genGivenVals<DTRes>(3, {3, 1, 2});
    
    checkAggRow(AggOpCode::IDXMAX, m0, m0exp);
    checkAggRow(AggOpCode::IDXMAX, m1, m1exp);
    checkAggRow(AggOpCode::IDXMAX, m2, m2exp);
    
    DataObjectFactory::destroy(m0, m0exp, m1, m1exp, m2, m2exp);
}


TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<typename DTArg::VT>;
    
    auto m0 = genGivenVals<DTArg>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0});
    auto m1 = genGivenVals<DTArg>(3, {
        5, 7, 3, 9,
        2, 5, 0, 1,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DTRes>(3, {6, 2, 5});
    auto m2 = genGivenVals<DTArg>(3, {
        4, 0, 0, 8,
        0, 4, 0, 0,
        0, 0, 8, 0,
    });
    auto m2exp = genGivenVals<DTRes>(3, {3, 1, 2});
    
    checkAggRow(AggOpCode::MEAN, m0, m0exp);
    checkAggRow(AggOpCode::MEAN, m1, m1exp);
    checkAggRow(AggOpCode::MEAN, m2, m2exp);
    
    DataObjectFactory::destroy(m0, m0exp, m1, m1exp, m2, m2exp);
}