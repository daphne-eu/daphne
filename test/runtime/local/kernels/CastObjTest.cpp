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
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/kernels/CastObj.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("castObj, frame to matrix, single-column", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DTRes = TestType;
    using VTRes = typename DTRes::VT;
    
    const size_t numRows = 4;
    auto c0 = genGivenVals<DenseMatrix<double>>(numRows, {0.0, 1.1, 2.2, 3.3});
    auto c0Exp = genGivenVals<DenseMatrix<VTRes>>(numRows, {VTRes(0.0), VTRes(1.1), VTRes(2.2), VTRes(3.3)});
    std::vector<Structure *> cols = {c0};
    auto arg = DataObjectFactory::create<Frame>(cols, nullptr);
    
    DTRes * res = nullptr;
    castObj<DTRes, Frame>(res, arg, nullptr);
    
    REQUIRE(res->getNumRows() == numRows);
    REQUIRE(res->getNumCols() == 1);
    CHECK(*res == *c0Exp);
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c0Exp);
    DataObjectFactory::destroy(arg);
    // TODO We cannot do this at the moment, due to a bug in sharing the
    // ownership of the underlying data (DenseMatrix should be constructed
    // from a std::shared_ptr in Frame::getColumn()).
//    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("castObj, frame to matrix, multi-column", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DTRes = TestType;
    using VTRes = typename DTRes::VT;
    
    const size_t numRows = 4;
    const size_t numCols = 3;
    auto c0 = genGivenVals<DenseMatrix<double>>(numRows, {0.0, 1.1, 2.2, 3.3});
    auto c1 = genGivenVals<DenseMatrix<int64_t>>(numRows, {0, -10, -20, -30});
    auto c2 = genGivenVals<DenseMatrix<uint8_t>>(numRows, {0, 11, 22, 33});
    auto c0Exp = genGivenVals<DenseMatrix<VTRes>>(numRows, {VTRes(0.0), VTRes(1.1), VTRes(2.2), VTRes(3.3)});
    auto c1Exp = genGivenVals<DenseMatrix<VTRes>>(numRows, {VTRes(0), VTRes(-10), VTRes(-20), VTRes(-30)});
    auto c2Exp = genGivenVals<DenseMatrix<VTRes>>(numRows, {VTRes(0), VTRes(11), VTRes(22), VTRes(33)});
    std::vector<Structure *> cols = {c0, c1, c2};
    auto arg = DataObjectFactory::create<Frame>(cols, nullptr);
    
    DTRes * res = nullptr;
    castObj<DTRes, Frame>(res, arg, nullptr);
    
    REQUIRE(res->getNumRows() == numRows);
    REQUIRE(res->getNumCols() == numCols);
    auto c0Fnd = DataObjectFactory::create<DTRes>(res, 0, numRows, 0, 1);
    auto c1Fnd = DataObjectFactory::create<DTRes>(res, 0, numRows, 1, 2);
    auto c2Fnd = DataObjectFactory::create<DTRes>(res, 0, numRows, 2, 3);
    CHECK(*c0Fnd == *c0Exp);
    CHECK(*c1Fnd == *c1Exp);
    CHECK(*c2Fnd == *c2Exp);
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(c0Exp);
    DataObjectFactory::destroy(c1Exp);
    DataObjectFactory::destroy(c2Exp);
    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(c0Fnd);
    DataObjectFactory::destroy(c1Fnd);
    DataObjectFactory::destroy(c2Fnd);
    DataObjectFactory::destroy(res);
}