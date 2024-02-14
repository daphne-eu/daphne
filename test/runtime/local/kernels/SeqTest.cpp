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
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/Seq.h>


#include <tags.h>
#include <catch.hpp>
#include <cstdint>

#define DATA_TYPES DenseMatrix
#define VALUE_TYPES  int32_t, int64_t 
#define VALUE_TYPES_FRAC float, double

template<class DT>
void checkSeq(DT *& res, typename DT::VT start, typename DT::VT end, typename DT::VT inc, DT * expectedMatrix) {
    seq<DT>(res, start, end, inc, nullptr);
    CHECK(*res == *expectedMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("Seq-basic-positive", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){

    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;

    targetMatrix = genGivenVals<DT>(5, {
        0,
        1,
        2,
        3,
        4,
    });
    checkSeq(inputMatrix,0,4, 1, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("Seq-reverse-positive", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;

    targetMatrix = genGivenVals<DT>(5, {
        4,
        3,
        2,
        1,
        0,
    });
    checkSeq(inputMatrix, 4, 0, -1, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}

 
TEMPLATE_PRODUCT_TEST_CASE("Seq-basic-negative", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;

    targetMatrix = genGivenVals<DT>(5, {
        0,
        -1,
        -2,
        -3,
        -4,
    });
    checkSeq(inputMatrix, 0, -4, -1, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("Seq-reverse-negative", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;

    targetMatrix = genGivenVals<DT>(5, {
        -4,
        -3,
        -2,
        -1,
        0,
    });
    checkSeq(inputMatrix, -4, 0, 1, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("Seq-basic-mix", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;

    targetMatrix = genGivenVals<DT>(5, {
        -4,
        -1,
        2,
        5,
        8,
    });
    checkSeq(inputMatrix, -4, 8, 3, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}


TEMPLATE_PRODUCT_TEST_CASE("Seq-reverse-mix", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;

    targetMatrix = genGivenVals<DT>(5, {
        8,
        5,
        2,
        -1,
        -4,
    });
    checkSeq(inputMatrix, 8, -4, -3, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);

}

TEMPLATE_PRODUCT_TEST_CASE("Seq-floating-step-forward", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES_FRAC)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;
    typename DT::VT start = 3.7;
    typename DT::VT end = 5.3;
    typename DT::VT inc= 0.4;
    typename DT::VT v1 = start + inc;
    typename DT::VT v2 = v1 + inc;
    typename DT::VT v3 = v2 + inc;
    typename DT::VT v4 = v3 + inc;

    targetMatrix = genGivenVals<DT>(5, {
        start,
        v1,
        v2,
        v3,
        v4,
    });
    checkSeq(inputMatrix, start, end, inc, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("Seq-floating-step-backward", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES_FRAC)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;
    typename DT::VT start = 5.3;
    typename DT::VT end = 3.7;
    typename DT::VT inc= -0.4;
    typename DT::VT v1 = start + inc;
    typename DT::VT v2 = v1 + inc;
    typename DT::VT v3 = v2 + inc;
    typename DT::VT v4 = v3 + inc;

    targetMatrix = genGivenVals<DT>(5, {
        start,
        v1,
        v2,
        v3,
        v4,
    });
    checkSeq(inputMatrix, start, end, inc, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}


TEMPLATE_PRODUCT_TEST_CASE("Seq-step>end", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES_FRAC)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;
    typename DT::VT start = 3.7;
    typename DT::VT end = 5.3;
    typename DT::VT inc= 5.3;

    targetMatrix = genGivenVals<DT>(1, {
        start,
    });
    checkSeq(inputMatrix, start, end, inc, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("Seq-end is not in the sequence", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES_FRAC)) {
    using DT = TestType;
    DT * inputMatrix = nullptr;
    DT * targetMatrix = nullptr;
    typename DT::VT start = 3.7;
    typename DT::VT end = 5;
    typename DT::VT inc = 0.4;
    typename DT::VT v1 = start + inc;
    typename DT::VT v2 = v1 + inc;
    typename DT::VT v3 = v2 + inc;

    targetMatrix = genGivenVals<DT>(4, {
        start,
        v1,
        v2,
        v3,
    });
    checkSeq(inputMatrix, start, end, inc, targetMatrix);
    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(inputMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("Seq-inc-does-not-lead-to-end", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    DT * res = nullptr;
    DT * targetMatrix = DataObjectFactory::create<DT>(0, 1, false);

    SECTION("positive inc") {
        checkSeq(res, 4, 0, 1, targetMatrix);
    }
    SECTION("negative inc") {
        checkSeq(res, 0, 4, -1, targetMatrix);
    }
    // TODO Test that zero increment yields an exception.

    DataObjectFactory::destroy(targetMatrix);
    DataObjectFactory::destroy(res);
}