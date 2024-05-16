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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <type_traits>
#include <vector>

#include <cstdint>

#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES double, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("CheckEq, original matrices", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 1, 0, 2, 0,
        0, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 6, 7,
        0, 8, 0, 0, 9, 0,
    };
    auto m1 = genGivenVals<DT>(4, vals);
    
    SECTION("same inst") {
        CHECK(*m1 == *m1);
    }
    SECTION("diff inst, same size, same cont") {
        auto m2 = genGivenVals<DT>(4, vals);
        CHECK(*m1 == *m2);
        DataObjectFactory::destroy(m2);
    }
    SECTION("diff inst, diff size, same cont") {
        auto m2 = genGivenVals<DT>(6, vals);
        CHECK_FALSE(*m1 == *m2);
        DataObjectFactory::destroy(m2);
    }
    SECTION("diff inst, same size, diff cont") {
        auto m2 = genGivenVals<DT>(4, {
            0, 0, 1, 0, 2, 0,
            0, 0, 1, 0, 0, 0,
            3, 4, 5, 0, 6, 7,
            0, 8, 0, 0, 9, 0,
        });
        CHECK_FALSE(*m1 == *m2);
        DataObjectFactory::destroy(m2);
    }
    SECTION("diff inst, diff size, diff cont") {
        auto m2 = genGivenVals<DT>(3, {
            1, 0, 0, 0,
            0, 2, 0, 4,
            0, 0, 3, 0,
        });
        CHECK_FALSE(*m1 == *m2);
        DataObjectFactory::destroy(m2);
    }

    DataObjectFactory::destroy(m1);
}
    
TEMPLATE_PRODUCT_TEST_CASE("CheckEq, views on matrices", TAG_KERNELS, (DenseMatrix, Matrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTGenView = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        DenseMatrix<VT>,
                        DT
                    >::type;
    
    std::vector<VT> vals = {
        1, 2, 2, 2, 0, 0,
        3, 4, 4, 4, 1, 2,
        0, 0, 0, 0, 3, 4,
        0, 0, 0, 0, 0, 0,
        1, 2, 0, 0, 0, 0,
        3, 4, 0, 0, 1, 2,
    };
    auto orig1 = genGivenVals<DTGenView>(6, vals);
    
    SECTION("same inst") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 0, 2));
        CHECK(*view1 == *view1);
        DataObjectFactory::destroy(view1);
    }
    SECTION("diff inst, same size, same cont, same orig") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 0, 2));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 0, 2));
        auto view3 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 1, 3, 4, 6));
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
        DataObjectFactory::destroy(view1, view2, view3);
    }
    SECTION("diff inst, same size, same cont, diff orig") {
        auto orig2 = genGivenVals<DTGenView>(6, vals);
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 0, 2));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig2, 0, 2, 0, 2));
        auto view3 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig2, 1, 3, 4, 6));
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
        DataObjectFactory::destroy(orig2, view1, view2, view3);
    }
    SECTION("diff inst, same size, same cont, overlap") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 1, 3));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 2, 4));
        CHECK(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }
    SECTION("diff inst, same size, diff cont") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 0, 2));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 4, 6, 4, 6));
        CHECK_FALSE(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }
    SECTION("diff inst, diff size, diff cont") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 2, 0, 2));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGenView>(orig1, 0, 3, 0, 2));
        CHECK_FALSE(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }

    DataObjectFactory::destroy(orig1);
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEq, views on matrices", TAG_KERNELS, (CSRMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 0, 0,
        0, 1, 0, 2,
        3, 0, 0, 0,
        0, 0, 4, 5,
        0, 0, 0, 0,
        3, 0, 0, 0,
        0, 0, 4, 5,
        0, 0, 4, 5,
        0, 0, 4, 5,
    };
    auto orig1 = genGivenVals<DT>(9, vals);
    
    SECTION("same inst") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 1, 4);
        CHECK(*view1 == *view1);
        DataObjectFactory::destroy(view1);
    }
    SECTION("diff inst, same size, same cont, same orig") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 2, 4);
        auto view2 = DataObjectFactory::create<DT>(orig1, 2, 4);
        auto view3 = DataObjectFactory::create<DT>(orig1, 5, 7);
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
        DataObjectFactory::destroy(view1, view2, view3);
    }
    SECTION("diff inst, same size, same cont, diff orig") {
        auto orig2 = genGivenVals<DT>(9, vals);
        auto view1 = DataObjectFactory::create<DT>(orig1, 2, 4);
        auto view2 = DataObjectFactory::create<DT>(orig2, 2, 4);
        auto view3 = DataObjectFactory::create<DT>(orig2, 5, 7);
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
        DataObjectFactory::destroy(orig2, view1, view2, view3);
    }
    SECTION("diff inst, same size, same cont, overlap") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 6, 8);
        auto view2 = DataObjectFactory::create<DT>(orig1, 7, 9);
        CHECK(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }
    SECTION("diff inst, same size, diff cont") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 3);
        auto view2 = DataObjectFactory::create<DT>(orig1, 3, 6);
        CHECK_FALSE(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }
    SECTION("diff inst, diff size, diff cont") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 3);
        auto view2 = DataObjectFactory::create<DT>(orig1, 3, 7);
        CHECK_FALSE(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }

    DataObjectFactory::destroy(orig1);
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEq, empty matrices", TAG_KERNELS, (DenseMatrix, Matrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTGen = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        DenseMatrix<VT>,
                        DT
                    >::type;
    
    std::vector<VT> vals = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    auto orig1 = genGivenVals<DTGen>(3, vals);
    
    SECTION("orig, diff inst, same size") {
        auto orig2 = genGivenVals<DTGen>(3, vals);
        CHECK(*orig1 == *orig2);
        DataObjectFactory::destroy(orig2);
    }
    SECTION("view, diff inst, same size") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGen>(orig1, 0, 2, 0, 4));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGen>(orig1, 1, 3, 0, 4));
        CHECK(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }
    SECTION("view, diff inst, diff size") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGen>(orig1, 0, 1, 0, 4));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGen>(orig1, 1, 3, 0, 4));
        CHECK_FALSE(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }

    DataObjectFactory::destroy(orig1);
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEq, empty matrices", TAG_KERNELS, (CSRMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    auto orig1 = genGivenVals<DT>(3, vals);
    
    SECTION("orig, diff inst, same size") {
        auto orig2 = genGivenVals<DT>(3, vals);
        CHECK(*orig1 == *orig2);
        DataObjectFactory::destroy(orig2);
    }
    SECTION("view, diff inst, same size") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2);
        auto view2 = DataObjectFactory::create<DT>(orig1, 1, 3);
        CHECK(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }
    SECTION("view, diff inst, diff size") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 1);
        auto view2 = DataObjectFactory::create<DT>(orig1, 1, 3);
        CHECK_FALSE(*view1 == *view2);
        DataObjectFactory::destroy(view1, view2);
    }

    DataObjectFactory::destroy(orig1);
}

TEST_CASE("CheckEq, frames", TAG_KERNELS) {
    using VT0 = int64_t;
    using VT1 = double;
    using VT2 = float;
    using VT3 = uint32_t;

    const size_t numRows = 4;

    auto c0 = genGivenVals<DenseMatrix<VT0>>(numRows, {VT0(0.0), VT0(1.1), VT0(2.2), VT0(3.3)});
    auto c1 = genGivenVals<DenseMatrix<VT1>>(numRows, {VT1(4.4), VT1(5.5), VT1(6.6), VT1(7.7)});
    auto c2 = genGivenVals<DenseMatrix<VT2>>(numRows, {VT2(8.8), VT2(9.9), VT2(1.0), VT2(2.0)});
   
    std::vector<Structure *> cols = {c0, c1, c2};
    auto frame1 = DataObjectFactory::create<Frame>(cols, nullptr);
    
    SECTION("same inst") {
        CHECK(*frame1 == *frame1);
    }
    SECTION("diff inst, same schema, same cont, no labels") {
        auto c3 = genGivenVals<DenseMatrix<VT2>>(numRows, {VT2(8.8), VT2(9.9), VT2(1.0), VT2(2.0)});
        std::vector<Structure *> cols2 = {c0, c1, c3};
        auto frame2 = DataObjectFactory::create<Frame>(cols2, nullptr);
        CHECK(*frame1 == *frame2);
        DataObjectFactory::destroy(frame2);
        DataObjectFactory::destroy(c3);
    }
    SECTION("diff inst, diff schema, same cont, no labels") {
        auto c3 = genGivenVals<DenseMatrix<VT3>>(numRows, {VT3(8.8), VT3(9.9), VT3(1.0), VT3(2.0)});
        std::vector<Structure *> cols2 = {c0, c1, c3};
        auto frame2 = DataObjectFactory::create<Frame>(cols2, nullptr);
        CHECK_FALSE(*frame1 == *frame2);
        DataObjectFactory::destroy(frame2);
        DataObjectFactory::destroy(c3);
    }
    SECTION("diff inst, same schema, diff cont, no labels") {
        auto c3 = genGivenVals<DenseMatrix<VT2>>(numRows, {VT2(8.0), VT2(0.9), VT2(1.0), VT2(0.2)});
        std::vector<Structure *> cols2 = {c0, c1, c3};
        auto frame2 = DataObjectFactory::create<Frame>(cols2, nullptr);
        CHECK_FALSE(*frame1 == *frame2);
        DataObjectFactory::destroy(frame2);
        DataObjectFactory::destroy(c3);
    }
    SECTION("diff inst, diff schema, diff cont, no labels") {
        auto c3 = genGivenVals<DenseMatrix<VT3>>(numRows, {VT3(8.0), VT3(0.9), VT3(1.0), VT3(0.2)});
        std::vector<Structure *> cols2 = {c0, c1, c3};
        auto frame2 = DataObjectFactory::create<Frame>(cols2, nullptr);
        CHECK_FALSE(*frame1 == *frame2);
        DataObjectFactory::destroy(frame2);
        DataObjectFactory::destroy(c3);
    }
    SECTION("diff inst, same schema, same cont, same labels") {
        auto c3 = genGivenVals<DenseMatrix<VT2>>(numRows, {VT2(8.8), VT2(9.9), VT2(1.0), VT2(2.0)});
        std::string * labels1 =  new std::string[3] {"ab", "cde", "fghi"};
        std::string * labels2 =  new std::string[3] {"ab", "cde", "fghi"};
        frame1 = DataObjectFactory::create<Frame>(cols, labels1);
        std::vector<Structure *> cols2 = {c0, c1, c3};
        auto frame2 = DataObjectFactory::create<Frame>(cols2, labels2);
        CHECK(*frame1 == *frame2);
        DataObjectFactory::destroy(frame2);
        DataObjectFactory::destroy(c3);
    }
    SECTION("diff inst, same schema, same cont, diff labels") {
        auto c3 = genGivenVals<DenseMatrix<VT2>>(numRows, {VT2(8.8), VT2(9.9), VT2(1.0), VT2(2.0)});
        std::string * labels1 =  new std::string[3] {"ab", "cde", "fghi"};
        std::string * labels2 =  new std::string[3] {"ab", "cde", "fxyz"};
        frame1 = DataObjectFactory::create<Frame>(cols, labels1);
        std::vector<Structure *> cols2 = {c0, c1, c3};
        auto frame2 = DataObjectFactory::create<Frame>(cols2, labels2);
        CHECK_FALSE(*frame1 == *frame2);
        DataObjectFactory::destroy(frame2);
        DataObjectFactory::destroy(c3);
    }

    DataObjectFactory::destroy(frame1, c0, c1, c2);
}