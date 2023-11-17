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
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0}); \
    auto m1 = genGivenVals<DTArg>(3, { \
        3, 0, 2, 0, \
        0, 0, 1, 1, \
        2, 5, 0, 0, \
    }); \
    auto m1exp = genGivenVals<DTRes>(3, {5, 2, 7}); \
     \
    checkAggRow(AggOpCode::SUM, m0, m0exp); \
    checkAggRow(AggOpCode::SUM, m1, m1exp); \
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

// The value types of argument and result can be assumed to be the same.
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

// The value type of the result can be assumed to be size_t.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("idxmin"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<size_t>;
    
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

// The value type of the result can be assumed to be size_t.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("idxmax"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = DenseMatrix<size_t>;
    
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

// The value types of argument and result could be different, so we need to
// test various combinations.
#define MEAN_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType; \
    using DTRes = DenseMatrix<VTRes>; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0}); \
    auto m1 = genGivenVals<DTArg>(3, { \
        5, 7, 3, 9, \
        2, 5, 0, 1, \
        7, 4, 5, 4, \
    }); \
    auto m1exp = genGivenVals<DTRes>(3, {6, 2, 5}); \
    auto m2 = genGivenVals<DTArg>(3, { \
        4, 0, 0, 8, \
        0, 4, 0, 0, \
        0, 0, 7, 0, \
    }); \
    auto m2exp = genGivenVals<DTRes>(3, {3, 1, (VTRes)1.75}); \
 \
    auto m3 = genGivenVals<DTArg>(3, { \
        5, 7, 1, 9, \
        2, 5, 7, 1, \
        7, 1, 5, 4, \
    }); \
    auto m3exp = genGivenVals<DTRes>(3, {(VTRes)5.5, (VTRes)3.75, (VTRes)4.25}); \
     \
    checkAggRow(AggOpCode::MEAN, m0, m0exp); \
    checkAggRow(AggOpCode::MEAN, m1, m1exp); \
    checkAggRow(AggOpCode::MEAN, m2, m2exp); \
    checkAggRow(AggOpCode::MEAN, m3, m3exp); \
     \
    DataObjectFactory::destroy(m0, m0exp, m1, m1exp, m2, m2exp, m3, m3exp); \
}
MEAN_TEST_CASE(int64_t);
MEAN_TEST_CASE(double);

#define STDDEV_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("stddev - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType; \
    using DTRes = DenseMatrix<VTRes>; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0}); \
    auto m1 = genGivenVals<DTArg>(3, { \
        4, 0, 0, 8, \
        0, 4, 0, 0, \
        0, 0, 7, 0, \
    }); \
    auto m1exp = genGivenVals<DTRes>(3, {(VTRes)3.3166247903553998491, (VTRes)1.7320508075688772935, (VTRes)3.0310889132455352637}); \
    auto m2 = genGivenVals<DTArg>(3, { \
        5, 7, 3, 9, \
        2, 5, 0, 1, \
        7, 4, 5, 4, \
    }); \
    auto m2exp = genGivenVals<DTRes>(3, {(VTRes)2.2360679774997896964, (VTRes)1.8708286933869706928, (VTRes)1.2247448713915890491}); \
     \
    checkAggRow(AggOpCode::STDDEV, m0, m0exp); \
    checkAggRow(AggOpCode::STDDEV, m1, m1exp); \
    checkAggRow(AggOpCode::STDDEV, m2, m2exp); \
     \
    DataObjectFactory::destroy(m0, m0exp); \
    DataObjectFactory::destroy(m1, m1exp); \
    DataObjectFactory::destroy(m2, m2exp); \
}
STDDEV_TEST_CASE(int64_t);
STDDEV_TEST_CASE(double);

#define VAR_TEST_CASE(VTRes) TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("var - result value type: " #VTRes), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) { \
    using DTArg = TestType; \
    using DTRes = DenseMatrix<VTRes>; \
     \
    auto m0 = genGivenVals<DTArg>(3, { \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
        0, 0, 0, 0, \
    }); \
    auto m0exp = genGivenVals<DTRes>(3, {0, 0, 0}); \
    auto m1 = genGivenVals<DTArg>(3, { \
        1, 1, 3, 3, \
        3, 3, 1, 1, \
        0, 5, 0, 5, \
    }); \
    auto m1exp = genGivenVals<DTRes>(3, {1, 1, (VTRes)6.25}); \
    auto m2 = genGivenVals<DTArg>(3, { \
        5, 7, 3, 9, \
        2, 5, 0, 1, \
        7, 4, 5, 4, \
    }); \
    auto m2exp = genGivenVals<DTRes>(3, {5, (VTRes)3.5, (VTRes)1.5}); \
     \
     \
    checkAggRow(AggOpCode::VAR, m0, m0exp); \
    checkAggRow(AggOpCode::VAR, m1, m1exp); \
    checkAggRow(AggOpCode::VAR, m2, m2exp); \
     \
    DataObjectFactory::destroy(m0, m0exp); \
    DataObjectFactory::destroy(m1, m1exp); \
    DataObjectFactory::destroy(m2, m2exp); \
}
VAR_TEST_CASE(int64_t);
VAR_TEST_CASE(double);