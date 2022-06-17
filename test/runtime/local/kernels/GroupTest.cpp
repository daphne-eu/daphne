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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/Order.h>
#include <runtime/local/kernels/Group.h>
#include <runtime/local/kernels/CheckEq.h>
#include <ir/daphneir/Daphne.h>

#include <tags.h>
#include <catch.hpp>
#include <vector>


TEMPLATE_TEST_CASE("Group", TAG_KERNELS, (Frame)) {
    using VT0 = double;
    using VT1 = float;
    using VT2 = int64_t;
    using VT3 = uint32_t;

    size_t numRows = 20;

    auto c0 = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5,
                                                        2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5 });
    auto c1 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.6, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.6, 2.8, 1.5,
                                                        2.7, 1.6, 2.7, 1.5, 2.8, 1.5, 2.7, 1.6, 2.8, 1.5 });
    auto c2 = genGivenVals<DenseMatrix<VT2>>(numRows, { -1, 0, 1, -1, 0, 1, -1, 0, 1, 3,
                                                        2, -1, 0, 1, 1, 1, 1, -1, -1, -1});
    auto c3 = genGivenVals<DenseMatrix<VT3>>(numRows, { 1, 0, 1, 1, 0, 1, 1, 0, 1, 3,
                                                        2, 1, 0, 1, 1, 1, 1, 1, 1, 1});;
    auto c4 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.5, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5,
                                                        2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5 });
    auto c5 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.6, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.6, 2.8, 1.5,
                                                        2.7, 1.6, 2.7, 1.5, 2.8, 1.5, 2.7, 1.6, 2.8, 1.5 });
    auto c6 = genGivenVals<DenseMatrix<VT2>>(numRows, { -1, 0, 1, -37, 17, 1, -1, 0, 1, 3,
                                                        2, -1, 0, 1, 1, 1, 1, -1, -1, -1});
    auto c7 = genGivenVals<DenseMatrix<VT2>>(numRows, { 1, 0, 1, 1, 0, 1, 1, 0, 1, 3,
                                                        2, 1, 0, 1, 1, 1, 1, 1, 1, 1});
    auto c8 = genGivenVals<DenseMatrix<VT3>>(numRows, { 1, 2, 3, 1, 2, 1, 2, 1, 2, 1,
                                                        2, 1, 2, 1, 2, 1, 2, 1, 2, 1 });
    auto c9 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.6, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.6, 2.8, 1.5,
                                                        2.7, 1.6, 2.7, 1.5, 2.8, 1.5, 2.7, 1.6, 2.8, 1.5 });
    
    std::vector<Structure *> colsArg {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9};
    std::string labels[] = {"aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "jjj"};
    auto arg = DataObjectFactory::create<Frame>(colsArg, labels);
    DataObjectFactory::destroy(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);

    Frame* exp{};
    Frame* res{};
    size_t numKeyCols;
    size_t numAggCols;
    const char** keyCols = nullptr;
    const char** aggCols = nullptr;;
    mlir::daphne::GroupEnum * aggFuncs = nullptr;
    
    std::vector<Structure *> colsExp;

    auto context = new mlir::MLIRContext();

    SECTION("1 grouping column, 1 aggregation column") {
        numKeyCols = 1;
        numAggCols = 1;
        keyCols = new const char*[10]{labels[0].c_str()};
        aggCols = new const char*[10]{labels[2].c_str()};
        aggFuncs = new mlir::daphne::GroupEnum[numAggCols];
        aggFuncs[0] = mlir::daphne::GroupEnum::COUNT;

        numRows = 3;
        DenseMatrix<VT0> * c0Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 2.7, 3.2 });
        DenseMatrix<uint64_t> * c1Exp = genGivenVals<DenseMatrix<uint64_t>>(numRows, { 10, 9, 1});
        std::vector<Structure *> colsExp {c0Exp, c1Exp};
        std::string labelsExp[] = {"aaa", "COUNT(ccc)"};
        exp = DataObjectFactory::create<Frame>(colsExp, labelsExp);
        DataObjectFactory::destroy(c0Exp, c1Exp);
    }
    SECTION("1 grouping column, 3 aggregation columns") {
        numKeyCols = 1;
        numAggCols = 3;
        keyCols = new const char*[10]{labels[7].c_str()};
        aggCols = new const char*[10]{labels[0].c_str(), labels[3].c_str(), labels[2].c_str()};
        aggFuncs = new mlir::daphne::GroupEnum[numAggCols];
        aggFuncs[0] = mlir::daphne::GroupEnum::SUM;
        aggFuncs[1] = mlir::daphne::GroupEnum::MIN;
        aggFuncs[2] = mlir::daphne::GroupEnum::MAX;

        numRows = 4;
        DenseMatrix<VT2> * c0Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { 0, 1, 2, 3 });
        DenseMatrix<VT0> * c1Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { 3*2.7+1.5, 8*1.5+5*2.7+3.2, 2.7, 1.5 });
        DenseMatrix<VT3> * c2Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 0, 1, 2, 3 });
        DenseMatrix<VT2> * c3Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { 0, 1, 2, 3 });

        std::vector<Structure *> colsExp {c0Exp, c1Exp, c2Exp, c3Exp};
        std::string labelsExp[] = {"hhh", "SUM(aaa)", "MIN(ddd)", "MAX(ccc)"};
        exp = DataObjectFactory::create<Frame>(colsExp, labelsExp);
        DataObjectFactory::destroy(c0Exp, c1Exp, c2Exp, c3Exp);
    }
    SECTION("3 grouping columns, 1 aggregation column") {
        numKeyCols = 3;
        numAggCols = 1;
        keyCols = new const char*[10]{labels[2].c_str(), labels[1].c_str(), labels[0].c_str()};
        aggCols = new const char*[10]{labels[6].c_str()};
        aggFuncs = new mlir::daphne::GroupEnum[numAggCols];
        aggFuncs[0] = mlir::daphne::GroupEnum::AVG;

        numRows = 12;
        DenseMatrix<VT2> * c0Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 2, 3 });
        DenseMatrix<VT1> * c1Exp = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.5, 1.6, 2.7, 2.8, 1.6, 2.7, 1.5, 2.7, 2.8, 3.2, 2.7, 1.5 });
        DenseMatrix<VT0> * c2Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 1.5, 2.7, 2.7, 1.5, 2.7, 1.5, 2.7, 2.7, 3.2, 2.7, 1.5 });
        DenseMatrix<double> * c3Exp = genGivenVals<DenseMatrix<double>>(numRows, { -19, -1, -1, -1, 0, 17.0/3.0, 1, 1, 1, 1, 2, 3 });
        std::vector<Structure *> colsExp {c0Exp, c1Exp, c2Exp, c3Exp};
        std::string labelsExp[] = {"ccc", "bbb", "aaa", "AVG(ggg)"};
        exp = DataObjectFactory::create<Frame>(colsExp, labelsExp);
        DataObjectFactory::destroy(c0Exp, c1Exp, c2Exp, c3Exp);
    }
    SECTION("5 grouping columns, 5 aggregation columns") {
        numKeyCols = 5;
        numAggCols = 5;
        keyCols = new const char*[10]{labels[0].c_str(), labels[6].c_str(), labels[2].c_str(), labels[8].c_str(), labels[4].c_str()};
        aggCols = new const char*[10]{labels[5].c_str(), labels[1].c_str(), labels[7].c_str(), labels[3].c_str(), labels[9].c_str()};
        aggFuncs = new mlir::daphne::GroupEnum[numAggCols];
        aggFuncs[0] = mlir::daphne::GroupEnum::COUNT;
        aggFuncs[1] = mlir::daphne::GroupEnum::SUM;
        aggFuncs[2] = mlir::daphne::GroupEnum::MIN;
        aggFuncs[3] = mlir::daphne::GroupEnum::MAX;
        aggFuncs[4] = mlir::daphne::GroupEnum::AVG;

        numRows = 11;
        DenseMatrix<VT0> * c0Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 1.5, 1.5, 1.5, 1.5, 2.7, 2.7, 2.7, 2.7, 2.7, 3.2 });
        DenseMatrix<VT2> * c1Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { -37, -1, 0, 1, 3, -1, 0, 1, 2, 17, 1 });
        DenseMatrix<VT2> * c2Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { -1, -1, 0, 1, 3, -1, 0, 1, 2, 0, 1 });
        DenseMatrix<VT3> * c3Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3 });
        DenseMatrix<VT1> * c4Exp = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.5, 1.5, 1.5, 1.5, 1.5, 2.7, 2.7, 2.7, 2.7, 2.7, 3.2 });
        DenseMatrix<uint64_t> * c5Exp = genGivenVals<DenseMatrix<uint64_t>>(numRows, { 1, 4, 1, 3, 1, 2, 2, 3, 1, 1, 1 });  
        DenseMatrix<VT1> * c6Exp = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.5f, 3*1.6f+1.5f, 1.6f, 3*1.5f, 1.5f, 2.7f+2.8f, 2*2.7f, 2*2.8f+2.7f, 2.7f, 2.7f, 3.2f });
        DenseMatrix<VT2> * c7Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { 1, 1, 0, 1, 3, 1, 0, 1, 2, 0, 1 });
        DenseMatrix<VT3> * c8Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 1, 1, 0, 1, 3, 1, 0, 1, 2, 0, 1 });
        DenseMatrix<VT0> * c9Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { (double)1.5f, (((double)3*1.6f)+((double)1.5f))/4.0, (double)1.6f, ((double)(3*1.5f))/3.0,
                                                                             (double)1.5f, ((double)(2.7f+2.8f))/2.0, ((double)(2*2.7f))/2.0, (((double)2*2.8f)+((double)2.7f))/3.0,
                                                                             (double)2.7f, (double)2.7f, ((double)3.2f) });
        std::vector<Structure *> colsExp {c0Exp, c1Exp, c2Exp, c3Exp, c4Exp, c5Exp, c6Exp, c7Exp, c8Exp, c9Exp};
        std::string labelsExp[] = {"aaa", "ggg", "ccc", "iii", "eee", "COUNT(fff)", "SUM(bbb)", "MIN(hhh)", "MAX(ddd)", "AVG(jjj)"};
        exp = DataObjectFactory::create<Frame>(colsExp, labelsExp);
        DataObjectFactory::destroy(c0Exp, c1Exp, c2Exp, c3Exp, c4Exp, c5Exp, c6Exp, c7Exp, c8Exp, c9Exp);
    }
    SECTION("0 grouping columns, 2 identical aggregation columns") {
        numKeyCols = 0;
        numAggCols = 2;
        keyCols = new const char*[10]{};
        aggCols = new const char*[10]{labels[2].c_str(), labels[2].c_str()};
        aggFuncs = new mlir::daphne::GroupEnum[numAggCols];
        aggFuncs[0] = mlir::daphne::GroupEnum::COUNT;
        aggFuncs[1] = mlir::daphne::GroupEnum::SUM;

        numRows = 1;
        DenseMatrix<uint64_t> * c0Exp = genGivenVals<DenseMatrix<uint64_t>>(numRows, { 20 });
        DenseMatrix<VT2> * c1Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { 5 });
        std::vector<Structure *> colsExp {c0Exp, c1Exp};
        std::string labelsExp[] = {"COUNT(ccc)", "SUM(ccc)"};
        exp = DataObjectFactory::create<Frame>(colsExp, labelsExp);
        DataObjectFactory::destroy(c0Exp, c1Exp);
    }
    SECTION("3 grouping column, 0 aggregation columns") {
        numKeyCols = 3;
        numAggCols = 0;
        keyCols = new const char*[10]{labels[0].c_str(), labels[2].c_str(), labels[3].c_str()};
        aggCols = new const char*[10]{};
        aggFuncs = nullptr;

        numRows = 9;
        DenseMatrix<VT0> * c0Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 1.5, 1.5, 1.5, 2.7, 2.7, 2.7, 2.7, 3.2 });
        DenseMatrix<VT2> * c1Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { -1, 0, 1, 3, -1, 0, 1, 2, 1 });
        DenseMatrix<VT3> * c2Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 1, 0, 1, 3, 1, 0, 1, 2, 1 });

        std::vector<Structure *> colsExp {c0Exp, c1Exp, c2Exp};
        std::string labelsExp[] = {"aaa", "ccc", "ddd"};
        exp = DataObjectFactory::create<Frame>(colsExp, labelsExp);
        DataObjectFactory::destroy(c0Exp, c1Exp, c2Exp);
    }

    group(res, arg, keyCols, numKeyCols, aggCols, numAggCols, aggFuncs, numAggCols, nullptr);
    CHECK(*res == *exp);
    delete [] keyCols;
    delete [] aggCols;
    delete aggFuncs;
    delete context;
    DataObjectFactory::destroy(arg, exp, res);
}