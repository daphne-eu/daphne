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
#include <runtime/local/kernels/CheckEqApprox.h>
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
    CHECK(checkEqApprox(res, exp, 1e-5, nullptr));
}

// The value types of argument and result could be different, so we need to
// test various combinations.
#define SUM_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType; \
    using DTRes = DenseMatrix<VTRes>; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0}); \
    auto m1 = genGivenVals<DTArg>(3, { \
        3, 0, 2, 0, \
        0, 0, 1, 1, \
        2, 5, 0, 0, \
    }); \
    auto m1exp = genGivenVals<DTRes>(1, {5, 5, 3, 1}); \
     \
    checkAggCol(AggOpCode::SUM, m0, m0exp); \
    checkAggCol(AggOpCode::SUM, m1, m1exp); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m0exp); \
    DataObjectFactory::destroy(m1); \
    DataObjectFactory::destroy(m1exp); \
}
SUM_TEST_CASE(int64_t)
SUM_TEST_CASE(double)

// The value types of argument and result can be assumed to be the same.
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
    
    checkAggCol(AggOpCode::MIN, m0, m0exp);
    checkAggCol(AggOpCode::MIN, m1, m1exp);
    checkAggCol(AggOpCode::MIN, m2, m2exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m2exp);
}

// The value types of argument and result can be assumed to be the same.
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
    
    checkAggCol(AggOpCode::MAX, m0, m0exp);
    checkAggCol(AggOpCode::MAX, m1, m1exp);
    checkAggCol(AggOpCode::MAX, m2, m2exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m2exp);
}

// The value types of argument and result could be different, so we need to
// test various combinations.
#define MEAN_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (int64_t, double)) { \
    using DTArg = TestType; \
    using DTRes = DenseMatrix<VTRes>; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0}); \
    auto m2 = genGivenVals<DTArg>(4, { \
        1, 3, 0, -1, \
        1, 3, 5,  3, \
        3, 1, 0,  0, \
        3, 1, 5, -1, \
    }); \
    auto m2exp = genGivenVals<DTRes>(1, {(VTRes)2.0, (VTRes)2.0, (VTRes)2.5, (VTRes)0.25}); \
    \
    checkAggCol(AggOpCode::MEAN, m0, m0exp); \
    checkAggCol(AggOpCode::MEAN, m2, m2exp); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m0exp); \
    DataObjectFactory::destroy(m2); \
    DataObjectFactory::destroy(m2exp); \
}
MEAN_TEST_CASE(int64_t);
MEAN_TEST_CASE(double);

// The value types of argument and result could be different, so we need to
// test various combinations.
#define STDDEV_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("stddev - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (int64_t, double)) { \
    using DTArg = TestType; \
    using DTRes = DenseMatrix<VTRes>; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0}); \
    auto m2 = genGivenVals<DTArg>(4, { \
        1, 3, 0, -1, \
        1, 3, 5,  3, \
        3, 1, 0,  0, \
        3, 1, 5, -1, \
    }); \
    auto m2exp = genGivenVals<DTRes>(1, {1, 1, (VTRes)2.5, (VTRes)1.6393596310755}); \
     \
    checkAggCol(AggOpCode::STDDEV, m0, m0exp); \
    checkAggCol(AggOpCode::STDDEV, m2, m2exp); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m0exp); \
    DataObjectFactory::destroy(m2); \
    DataObjectFactory::destroy(m2exp); \
}
STDDEV_TEST_CASE(int64_t);
STDDEV_TEST_CASE(double);

#define VAR_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("var - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (int64_t, double)) { \
    using DTArg = TestType; \
    using DTRes = DenseMatrix<VTRes>; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m0exp = genGivenVals<DTRes>(1, {0, 0, 0, 0}); \
    auto m1 = genGivenVals<DTArg>(4, { \
        1, 3, 0, \
        1, 3, 5, \
        3, 1, 0, \
        3, 1, 5, \
    }); \
    auto m1exp = genGivenVals<DTRes>(1, {1, 1, (VTRes)6.25}); \
     \
    checkAggCol(AggOpCode::VAR, m0, m0exp); \
    checkAggCol(AggOpCode::VAR, m1, m1exp); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m0exp); \
    DataObjectFactory::destroy(m1); \
    DataObjectFactory::destroy(m1exp); \
}
VAR_TEST_CASE(int64_t);
VAR_TEST_CASE(double);