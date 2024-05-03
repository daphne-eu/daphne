/*
 * Copyright 2023 The DAPHNE Consortium
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
#include <runtime/local/kernels/AggCum.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TEST_NAME(opName) "AggCum (" opName ")"
#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES double, int32_t

template<class DTRes, class DTArg>
void checkAggCum(AggOpCode opCode, const DTArg * arg, const DTRes * exp) {
    DTRes * res = nullptr;
    aggCum<DTRes, DTArg>(opCode, res, arg, nullptr);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = TestType;
    using DTEmpty = DenseMatrix<typename DTArg::VT>;
    
    DTArg * arg = nullptr;
    DTRes * exp = nullptr;

    SECTION("0x0 matrix") {
        // can't create an empty generic Matrix, so DenseMatrix is casted instead
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
        exp = static_cast<DTRes *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
    }
    SECTION("1xn matrix") {
        arg = genGivenVals<DTArg>(1, {1, -2, 3});
        exp = genGivenVals<DTArg>(1, {1, -2, 3});
    }
    SECTION("mx1 matrix") {
        arg = genGivenVals<DTArg>(3, {1, -2, 3});
        exp = genGivenVals<DTArg>(3, {1, -1, 2});
    }
    SECTION("mxn matrix, zero") {
        arg = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
    }
    SECTION("mxn matrix, sparse") {
        arg = genGivenVals<DTArg>(4, {
            0, 0,  1,
            0, 2,  0,
            0, 0, -3,
            0, 0,  0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0,  1,
            0, 2,  1,
            0, 2, -2,
            0, 2, -2,
        });
    }
    SECTION("mxn matrix, dense") {
        arg = genGivenVals<DTArg>(4, {
             4, -2, -2,
            -3,  3,  1,
             1,  4, -5,
             5,  2,  6,
        });
        exp = genGivenVals<DTArg>(4, {
            4, -2, -2,
            1,  1, -1,
            2,  5, -6,
            7,  7,  0,
        });
    }

    checkAggCum(AggOpCode::SUM, arg, exp);
    
    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("prod"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = TestType;
    using DTEmpty = DenseMatrix<typename DTArg::VT>;

    DTArg * arg = nullptr;
    DTRes * exp = nullptr;

    SECTION("0x0 matrix") {
        // can't create an empty generic Matrix, so DenseMatrix is casted instead
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
        exp = static_cast<DTRes *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
    }
    SECTION("1xn matrix") {
        arg = genGivenVals<DTArg>(1, {1, -2, 3});
        exp = genGivenVals<DTArg>(1, {1, -2, 3});
    }
    SECTION("mx1 matrix") {
        arg = genGivenVals<DTArg>(3, {1, -2,  3});
        exp = genGivenVals<DTArg>(3, {1, -2, -6});
    }
    SECTION("mxn matrix, zero") {
        arg = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
    }
    SECTION("mxn matrix, sparse") {
        arg = genGivenVals<DTArg>(4, {
            0, 0, 1,
            0, 2, 0,
            0, 0, 3,
            0, 0, 0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0, 1,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
    }
    SECTION("mxn matrix, dense") {
        arg = genGivenVals<DTArg>(4, {
             4, -2, -2,
            -3,  3,  1,
             1,  4, -5,
             5,  2,  6,
        });
        exp = genGivenVals<DTArg>(4, {
              4,  -2, -2,
            -12,  -6, -2,
            -12, -24, 10,
            -60, -48, 60,
        });
    }

    checkAggCum(AggOpCode::PROD, arg, exp);
    
    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = TestType;
    using DTEmpty = DenseMatrix<typename DTArg::VT>;
    
    DTArg * arg = nullptr;
    DTRes * exp = nullptr;

    SECTION("0x0 matrix") {
        // can't create an empty generic Matrix, so DenseMatrix is casted instead
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
        exp = static_cast<DTRes *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
    }
    SECTION("1xn matrix") {
        arg = genGivenVals<DTArg>(1, {1, -2, 3});
        exp = genGivenVals<DTArg>(1, {1, -2, 3});
    }
    SECTION("mx1 matrix") {
        arg = genGivenVals<DTArg>(3, {1, -2,  3});
        exp = genGivenVals<DTArg>(3, {1, -2, -2});
    }
    SECTION("mxn matrix, zero") {
        arg = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
    }
    SECTION("mxn matrix, sparse") {
        arg = genGivenVals<DTArg>(4, {
            0, 0,  1,
            0, 2,  0,
            0, 0, -3,
            0, 0,  0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0,  1,
            0, 0,  0,
            0, 0, -3,
            0, 0, -3,
        });
    }
    SECTION("mxn matrix, dense") {
        arg = genGivenVals<DTArg>(4, {
             4, -2, -2,
            -3,  3,  1,
             1,  4, -5,
             5,  2,  6,
        });
        exp = genGivenVals<DTArg>(4, {
             4, -2, -2,
            -3, -2, -2,
            -3, -2, -5,
            -3, -2, -5,
        });
    }

    checkAggCum(AggOpCode::MIN, arg, exp);
    
    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using DTRes = TestType;
    using DTEmpty = DenseMatrix<typename DTArg::VT>;
    
    DTArg * arg = nullptr;
    DTRes * exp = nullptr;

    SECTION("0x0 matrix") {
        // can't create an empty generic Matrix, so DenseMatrix is casted instead
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
        exp = static_cast<DTRes *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
    }
    SECTION("1xn matrix") {
        arg = genGivenVals<DTArg>(1, {1, -2, 3});
        exp = genGivenVals<DTArg>(1, {1, -2, 3});
    }
    SECTION("mx1 matrix") {
        arg = genGivenVals<DTArg>(3, {1, -2, 3});
        exp = genGivenVals<DTArg>(3, {1,  1, 3});
    }
    SECTION("mxn matrix, zero") {
        arg = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
    }
    SECTION("mxn matrix, sparse") {
        arg = genGivenVals<DTArg>(4, {
            0, 0,  1,
            0, 2,  0,
            0, 0, -3,
            0, 0,  0,
        });
        exp = genGivenVals<DTArg>(4, {
            0, 0, 1,
            0, 2, 1,
            0, 2, 1,
            0, 2, 1,
        });
    }
    SECTION("mxn matrix, dense") {
        arg = genGivenVals<DTArg>(4, {
             4, -2, -2,
            -3,  3,  1,
             1,  4, -5,
             5,  2,  6,
        });
        exp = genGivenVals<DTArg>(4, {
            4, -2, -2,
            4,  3,  1,
            4,  4,  1,
            5,  4,  6,
        });
    }

    checkAggCum(AggOpCode::MAX, arg, exp);
    
    DataObjectFactory::destroy(arg, exp);
}