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

    GroupEnum aggFuncArray[] = { GroupEnum::COUNT, GroupEnum::SUM, GroupEnum::MIN, GroupEnum::MAX, GroupEnum::AVG };
    size_t numRows = 20;

    auto c0 = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5,
                                                        2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5 });
    auto c1 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.6, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.6, 2.8, 1.5,
                                                        2.7, 1.6, 2.7, 1.5, 2.8, 1.5, 2.7, 1.6, 2.8, 1.5 });
    auto c2 = genGivenVals<DenseMatrix<VT2>>(numRows, { -1, 0, 1, -1, 0, 1, -1, 0, 1, 3,
                                                        2, -1, 0, 1, 1, 1, 1, -1, -1, -1});
    auto c3 = genGivenVals<DenseMatrix<VT3>>(numRows, { 1, 0, 1, 1, 0, 1, 1, 0, 1, 3,
                                                        2, 1, 0, 1, 1, 1, 1, 1, 1, 1});;
    auto c4 = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5,
                                                        2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5 });
    auto c5 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.6, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.6, 2.8, 1.5,
                                                        2.7, 1.6, 2.7, 1.5, 2.8, 1.5, 2.7, 1.6, 2.8, 1.5 });
    auto c6 = genGivenVals<DenseMatrix<VT2>>(numRows, { -1, 0, 1, -1, 0, 1, -1, 0, 1, 3,
                                                        2, -1, 0, 1, 1, 1, 1, -1, -1, -1});
    auto c7 = genGivenVals<DenseMatrix<VT3>>(numRows, { 1, 0, 1, 1, 0, 1, 1, 0, 1, 3,
                                                        2, 1, 0, 1, 1, 1, 1, 1, 1, 1});
    auto c8 = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.5, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5,
                                                        2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5, 2.7, 1.5 });
    auto c9 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.6, 2.7, 3.2, 1.5, 2.7, 1.5, 2.7, 1.6, 2.8, 1.5,
                                                        2.7, 1.6, 2.7, 1.5, 2.8, 1.5, 2.7, 1.6, 2.8, 1.5 });
    
    std::vector<Structure *> colsArg {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9};
    std::string labels[] = {"aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "jjj",
                            "kkk", "lll", "mmm", "nnn", "ooo", "ppp", "qqq", "rrr", "sss", "ttt",};
    auto arg = DataObjectFactory::create<Frame>(colsArg, labels);
    Frame* exp{};
    Frame* res{};
    size_t numKeyCols;
    size_t numAggCols;
    const char** keyCols;
    const char** aggCols;
    GroupEnumAttr * aggFuncs;
    
    std::vector<Structure *> colsExp;
    std::string * labelsExp;

    auto context = new mlir::MLIRContext();

    SECTION("1 grouping column, 1 aggregation column") {
        numKeyCols = 1;
        numAggCols = 1;
        keyCols = new const char*[10]{labels[0].c_str()};
        aggCols = new const char*[10]{labels[2].c_str()};
        aggFuncs = new mlir::daphne::GroupEnumAttr[numAggCols];
        aggFuncs[0] = mlir::daphne::GroupEnumAttr::get(context, mlir::daphne::GroupEnum::COUNT);

        DenseMatrix<VT0> * c0Exp = genGivenVals<DenseMatrix<VT0>>(3, { 1.5, 2.7, 3.2 });
        DenseMatrix<uint64_t> * c1Exp = genGivenVals<DenseMatrix<uint64_t>>(3, { 10, 9, 1});
        std::vector<Structure *> colsExp {c0Exp, c1Exp};
        std::string labelsExp[] = {"aaa", "COUNT(ccc)"};
        exp = DataObjectFactory::create<Frame>(colsExp, labelsExp);
        DataObjectFactory::destroy(c0Exp, c1Exp);
    }
    //TODO: more testcases (multi group combinations: 1-3 SUM, MIN, MAX; 3-1 AVG; 5-5 all functions)

    DataObjectFactory::destroy(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
    group(res, arg, keyCols, numKeyCols, aggCols, numAggCols, aggFuncs, numAggCols, nullptr);
    CHECK(*res == *exp);

    delete [] keyCols;
    delete [] aggCols;
    delete aggFuncs;
    DataObjectFactory::destroy(arg, exp, res);
}