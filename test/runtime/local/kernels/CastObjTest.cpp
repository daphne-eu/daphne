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
    DataObjectFactory::destroy(res);
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

TEMPLATE_PRODUCT_TEST_CASE("castObj, matrix to frame, single-column", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DTArg = TestType;
    using VTArg = typename DTArg::VT;

    const size_t numRows = 4;
    auto arg = genGivenVals<DenseMatrix<VTArg>>(numRows,{VTArg(0.0), VTArg(1.1), VTArg(2.2), VTArg(3.3),});    
    std::vector<Structure *> cols = {arg};
    auto exp = DataObjectFactory::create<Frame>(cols, nullptr);

    Frame * res = nullptr;
    castObj<Frame, DTArg>(res, arg, nullptr);
    CHECK(*res == *exp);

    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("castObj, matrix to frame, multi-column", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DTArg = TestType;
    using VTArg = typename DTArg::VT;

    const size_t numRows = 4;
    auto arg = genGivenVals<DenseMatrix<VTArg>>(numRows, {
        VTArg(0.0), VTArg(1.1), VTArg(2.2),
        VTArg(3.3), VTArg(4.4), VTArg(5.5),
        VTArg(6.6), VTArg(7.7), VTArg(8.8),
        VTArg(9.9), VTArg(1.0), VTArg(2.0)
        });
    
    auto c0 = genGivenVals<DenseMatrix<VTArg>>(numRows, {VTArg(0.0), VTArg(3.3), VTArg(6.6), VTArg(9.9)});
    auto c1 = genGivenVals<DenseMatrix<VTArg>>(numRows, {VTArg(1.1), VTArg(4.4), VTArg(7.7), VTArg(1.0)});
    auto c2 = genGivenVals<DenseMatrix<VTArg>>(numRows, {VTArg(2.2), VTArg(5.5), VTArg(8.8), VTArg(2.0)});
    std::vector<Structure *> cols = {c0, c1, c2};
    auto exp = DataObjectFactory::create<Frame>(cols, nullptr);
    
    Frame * res = nullptr;
    castObj<Frame, DTArg>(res, arg, nullptr);
    CHECK(*res == *exp);

    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("castObj, matrix to frame and back, multi-column", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, {
            2, 3, 1, 1, 2, 4,
            1, 2, 1, 3, 1, 4,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
    });

    Frame * f0 = nullptr;
    castObj<Frame, DT>(f0, m0, nullptr);
    Frame * f1 = nullptr;
    castObj<Frame, DT>(f1, m1, nullptr);
    Frame * f2 = nullptr;
    castObj<Frame, DT>(f2, m2, nullptr);
    DT* res0 = nullptr;
    castObj<DT, Frame>(res0, f0, nullptr);
    DT* res1 = nullptr;
    castObj<DT, Frame>(res1, f1, nullptr);
    DT* res2 = nullptr;
    castObj<DT, Frame>(res2, f2, nullptr);
    CHECK(*m0 == *res0);
    CHECK(*m1 == *res1);
    CHECK(*m2 == *res2);

    DataObjectFactory::destroy(m0, f0, res0);
    DataObjectFactory::destroy(m1, f1, res1);
    DataObjectFactory::destroy(m2, f2, res2);
}

TEMPLATE_PRODUCT_TEST_CASE("castObj, matrix to matrix, multi-column", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DTRes = TestType;
    using VTRes = typename DTRes::VT;

    const size_t numRows = 2;

    auto arg1 = genGivenVals<DenseMatrix<double>>(numRows, {3., 1., 4., 1., 5., 9.});
    DTRes* res1 = nullptr;

    auto arg2 = genGivenVals<DenseMatrix<int64_t>>(numRows, {3, 1, 4, 1, 5, 9});
    DTRes* res2 = nullptr;

    auto arg3 = genGivenVals<DenseMatrix<uint32_t>>(numRows, {3, 1, 4, 1, 5, 9}); 
    DTRes* res3 = nullptr;
    
    auto check123 = genGivenVals<DenseMatrix<VTRes>>(numRows, {VTRes(3.), VTRes(1.), VTRes(4.), VTRes(1.), VTRes(5.), VTRes(9.)});

    castObj<DenseMatrix<VTRes>, DenseMatrix<double>>(res1, arg1, nullptr);
    castObj<DenseMatrix<VTRes>, DenseMatrix<int64_t>>(res2, arg2, nullptr);
    castObj<DenseMatrix<VTRes>, DenseMatrix<uint32_t>>(res3, arg3, nullptr);


    CHECK(*res1 == *check123);
    CHECK(*res2 == *check123);
    CHECK(*res3 == *check123);

    DataObjectFactory::destroy(arg1,arg2,arg3);
    DataObjectFactory::destroy(res1,res2,res3);
    DataObjectFactory::destroy(check123);
}

TEMPLATE_PRODUCT_TEST_CASE("castObj, matrix to matrix, single dim", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DTRes = TestType;
    using VTRes = typename DTRes::VT;


    // Single col
    size_t numRows = 3;

    auto arg1 = genGivenVals<DenseMatrix<double>>(numRows, {3., 1., 4.});
    DTRes* res1 = nullptr;

    auto arg2 = genGivenVals<DenseMatrix<int64_t>>(numRows, {3, 1, 4});
    DTRes* res2 = nullptr;

    auto arg3 = genGivenVals<DenseMatrix<uint32_t>>(numRows, {3, 1, 4});
    DTRes* res3 = nullptr; 
    
    auto check123 = genGivenVals<DenseMatrix<VTRes>>(numRows, {VTRes(3.), VTRes(1.), VTRes(4.)});

    castObj<DenseMatrix<VTRes>, DenseMatrix<double>>(res1, arg1, nullptr);
    castObj<DenseMatrix<VTRes>, DenseMatrix<int64_t>>(res2, arg2, nullptr);
    castObj<DenseMatrix<VTRes>, DenseMatrix<uint32_t>>(res3, arg3, nullptr);

    CHECK(*res1 == *check123);
    CHECK(*res2 == *check123);
    CHECK(*res3 == *check123);

    // Single row
    numRows = 1;

    auto arg4 = genGivenVals<DenseMatrix<double>>(numRows, {3., 1., 4.});
    DTRes* res4 = nullptr;

    auto arg5 = genGivenVals<DenseMatrix<int64_t>>(numRows, {3, 1, 4});
    DTRes* res5 = nullptr;

    auto arg6 = genGivenVals<DenseMatrix<uint32_t>>(numRows, {3, 1, 4});
    DTRes* res6 = nullptr;
    
    auto check456 = genGivenVals<DenseMatrix<VTRes>>(numRows, {VTRes(3.), VTRes(1.), VTRes(4.)});

    castObj<DenseMatrix<VTRes>, DenseMatrix<double>>(res4, arg4, nullptr);
    castObj<DenseMatrix<VTRes>, DenseMatrix<int64_t>>(res5, arg5, nullptr);
    castObj<DenseMatrix<VTRes>, DenseMatrix<uint32_t>>(res6, arg6, nullptr);

    CHECK(*res4 == *check456);
    CHECK(*res5 == *check456);
    CHECK(*res6 == *check456);
    
    DataObjectFactory::destroy(arg1,arg2,arg3, arg4,arg5,arg6);
    DataObjectFactory::destroy(res1, res2, res3, res4, res5, res6);
    DataObjectFactory::destroy(check123, check456);
}

TEMPLATE_PRODUCT_TEST_CASE("castObj, matrix to matrix, zero dim & dim mismatch", TAG_KERNELS, (DenseMatrix), (double, int64_t, uint32_t)) {
    using DTRes = TestType;
    using VTRes = typename DTRes::VT;

    // Zero dim
    size_t numRows = 0;
    
    size_t numCols = 0;
    auto arg1 =  DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);
    DTRes* res1 = nullptr;
    auto check1 = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

    numCols = 1;
    auto arg2 =  DataObjectFactory::create<DenseMatrix<int64_t>>(numRows, numCols, false);
    DTRes* res2 = nullptr;
    auto check2 = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

    castObj<DenseMatrix<VTRes>, DenseMatrix<double>>(res1, arg1, nullptr);
    castObj<DenseMatrix<VTRes>, DenseMatrix<int64_t>>(res2, arg2, nullptr);

    CHECK(*res1 == *check1);
    CHECK(*res2 == *check2);

    DataObjectFactory::destroy(arg1, arg2);
    DataObjectFactory::destroy(res1, res2);
    DataObjectFactory::destroy(check1, check2);
}

TEMPLATE_TEST_CASE("CastObj CSRMatrix to DenseMatrix", TAG_KERNELS, double, float, int64_t) {
    using VT = TestType;
    using DTArg = CSRMatrix<VT>;
    using DTRes= DenseMatrix<VT>;
    
    auto m0 = genGivenVals<DTArg>(4, {
            0, 0, 0, 0, 0, 0,
            0, 4, 0, 0, 0, 2,
            0, 0, 0, 3, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DTArg>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DTArg>(4, {
            2, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 4,
            0, 0, 0, 0, 1, 1,
            1, 0, 0, 0, 0, 1,
    });

    auto d0 = genGivenVals<DTRes>(4, {
            0, 0, 0, 0, 0, 0,
            0, 4, 0, 0, 0, 2,
            0, 0, 0, 3, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto d1 = genGivenVals<DTRes>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto d2 = genGivenVals<DTRes>(4, {
            2, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 4,
            0, 0, 0, 0, 1, 1,
            1, 0, 0, 0, 0, 1,
    });

    DTRes * res0 = nullptr;
    castObj<DTRes, DTArg>(res0, m0, nullptr);
    DTRes * res1 = nullptr;
    castObj<DTRes, DTArg>(res1, m1, nullptr);
    DTRes * res2 = nullptr;
    castObj<DTRes, DTArg>(res2, m2, nullptr);
    
    CHECK(*d0 == *res0);
    CHECK(*d1 == *res1);
    CHECK(*d2 == *res2);

    DataObjectFactory::destroy(m0, d0, res0);
    DataObjectFactory::destroy(m1, d1, res1);
    DataObjectFactory::destroy(m2, d2, res2);
}

TEMPLATE_TEST_CASE("CastObj DenseMatrix to CSRMatrix", TAG_KERNELS, double, float, int64_t) {
    using VT = TestType;
    using DTRes = CSRMatrix<VT>;
    using DTArg= DenseMatrix<VT>;
    
    auto m0 = genGivenVals<DTArg>(4, {
            0, 0, 0, 0, 0, 0,
            0, 4, 0, 0, 0, 2,
            0, 0, 0, 3, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DTArg>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DTArg>(4, {
            2, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 4,
            0, 0, 0, 0, 1, 1,
            1, 0, 0, 0, 0, 1,
    });

    auto d0 = genGivenVals<DTRes>(4, {
            0, 0, 0, 0, 0, 0,
            0, 4, 0, 0, 0, 2,
            0, 0, 0, 3, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto d1 = genGivenVals<DTRes>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto d2 = genGivenVals<DTRes>(4, {
            2, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 4,
            0, 0, 0, 0, 1, 1,
            1, 0, 0, 0, 0, 1,
    });

    DTRes * res0 = nullptr;
    castObj<DTRes, DTArg>(res0, m0, nullptr);
    DTRes * res1 = nullptr;
    castObj<DTRes, DTArg>(res1, m1, nullptr);
    DTRes * res2 = nullptr;
    castObj<DTRes, DTArg>(res2, m2, nullptr);
    
    CHECK(*d0 == *res0);
    CHECK(*d1 == *res1);
    CHECK(*d2 == *res2);

    DataObjectFactory::destroy(m0, d0, res0);
    DataObjectFactory::destroy(m1, d1, res1);
    DataObjectFactory::destroy(m2, d2, res2);
}