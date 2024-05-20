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
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TEST_NAME(opName) "AggAll (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES double, uint32_t

template<typename VTRes, class DTArg>
void checkAggAll(AggOpCode opCode, const DTArg * arg, VTRes exp) {
    VTRes res = aggAll<VTRes, DTArg>(opCode, arg, nullptr);
    CHECK(Approx(res).epsilon(1e-5) == exp);
}

// The value types of argument and result could be different, so we need to
// test various combinations.
#define SUM_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m1 = genGivenVals<DTArg>(3, { \
        3, 0, 2, 0, \
        0, 0, 1, 1, \
        2, 5, 0, 0, \
    }); \
     \
    checkAggAll(AggOpCode::SUM, m0, (VTRes)0); \
    checkAggAll(AggOpCode::SUM, m1, (VTRes)14); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m1); \
}
SUM_TEST_CASE(int64_t)
SUM_TEST_CASE(double)

// The value types of argument and result can be assumed to be the same.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m2 = genGivenVals<DT>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    
    // In case of min the result type is the same as the input type
    checkAggAll(AggOpCode::MIN, m0, (typename DT::VT)0);
    checkAggAll(AggOpCode::MIN, m1, (typename DT::VT)2);
    checkAggAll(AggOpCode::MIN, m2, (typename DT::VT)0);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// The value types of argument and result can be assumed to be the same.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m2 = genGivenVals<DT>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });

    // In case of max the result type is the same as the input type
    checkAggAll(AggOpCode::MAX, m0, (typename DT::VT)0);
    checkAggAll(AggOpCode::MAX, m1, (typename DT::VT)9);
    checkAggAll(AggOpCode::MAX, m2, (typename DT::VT)9);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// The value types of argument and result could be different, so we need to
// test various combinations.
#define MEAN_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType;  \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m1 = genGivenVals<DTArg>(3, { \
        1, 6, 3, 9, \
        2, 2, 8, 9, \
        4, 4, 5, 4, \
    }); \
    auto m2 = genGivenVals<DTArg>(3, { \
        4, 0, 0, 9, \
        0, 6, 0, 0, \
        0, 0, 5, 0, \
    }); \
     \
    checkAggAll(AggOpCode::MEAN, m0, (VTRes)0); \
    checkAggAll(AggOpCode::MEAN, m1, (VTRes)4.75); \
    checkAggAll(AggOpCode::MEAN, m2, (VTRes)2); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m1); \
    DataObjectFactory::destroy(m2); \
}
MEAN_TEST_CASE(int64_t);
MEAN_TEST_CASE(double);


#define STDDEV_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("stddev - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType;  \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m1 = genGivenVals<DTArg>(3, { \
        4, 0, 0, 9, \
        0, 6, 0, 0, \
        0, 0, 5, 0, \
    }); \
    auto m2 = genGivenVals<DTArg>(3, { \
        1, 6, 3, 9, \
        2, 2, 8, 9, \
        4, 4, 5, 4, \
    }); \
     \
    checkAggAll(AggOpCode::STDDEV, m0, (VTRes)0); \
    checkAggAll(AggOpCode::STDDEV, m1, (VTRes)3.0276503540974916654); \
    checkAggAll(AggOpCode::STDDEV, m2, (VTRes)2.6180463454008346998); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m1); \
    DataObjectFactory::destroy(m2); \
}
STDDEV_TEST_CASE(int64_t);
STDDEV_TEST_CASE(double);

#define VAR_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("var - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType;  \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m1 = genGivenVals<DTArg>(3, { \
        4, 0, 0, 9, \
        0, 6, 0, 0, \
        0, 0, 5, 0, \
    }); \
    auto m2 = genGivenVals<DTArg>(3, { \
        0, 1, 2, \
        4, 4, 5, \
        9, 12, 8, \
    }); \
     \
    checkAggAll(AggOpCode::VAR, m0, (VTRes)0); \
    checkAggAll(AggOpCode::VAR, m1, (VTRes)9.1666666666666666667); \
    checkAggAll(AggOpCode::VAR, m2, (VTRes)14); \
     \
    DataObjectFactory::destroy(m0); \
    DataObjectFactory::destroy(m1); \
    DataObjectFactory::destroy(m2); \
}
VAR_TEST_CASE(int64_t);
VAR_TEST_CASE(double);