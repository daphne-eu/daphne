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
#include <runtime/local/kernels/InsertRow.h>

#include <catch.hpp>
#include <tags.h>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES int32_t, double

template <typename DTRes, typename DTArg, typename DTIns, typename VTSel>
void checkInsertRow(const DTArg *arg, const DTIns *ins, const VTSel lowerIncl, const VTSel upperExcl,
                    const DTRes *exp) {
    DTRes *res = nullptr;
    insertRow<DTRes, DTArg, DTIns, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res, exp);
}

template <typename DTRes, typename DTArg, typename DTIns, typename VTSel>
void checkInsertRowThrow(const DTArg *arg, const DTIns *ins, const VTSel lowerIncl, const VTSel upperExcl) {
    DTRes *res = nullptr;
    REQUIRE_THROWS_AS((insertRow<DTRes, DTArg, DTIns, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr)),
                      std::out_of_range);
}

// Helper classes for combinations of data types.
template <typename VT> struct DenseMatrix_DenseMatrix_DenseMatrix {
    using DTRes = DenseMatrix<VT>;
    using DTArg = DenseMatrix<VT>;
    using DTIns = DenseMatrix<VT>;
};
template <typename VT> struct DenseMatrix_CSRMatric_DenseMatrix {
    using DTRes = DenseMatrix<VT>;
    using DTArg = CSRMatrix<VT>;
    using DTIns = DenseMatrix<VT>;
};
template <typename VT> struct Matrix_Matrix_Matrix {
    using DTRes = Matrix<VT>;
    using DTArg = Matrix<VT>;
    using DTIns = Matrix<VT>;
};
#define DATA_TYPE_COMBINATIONS                                                                                         \
    DenseMatrix_DenseMatrix_DenseMatrix, DenseMatrix_CSRMatric_DenseMatrix, Matrix_Matrix_Matrix

TEMPLATE_PRODUCT_TEST_CASE("InsertRow - valid", TAG_KERNELS, (DATA_TYPE_COMBINATIONS), (VALUE_TYPES)) {
    using DTRes = typename TestType::DTRes;
    using DTArg = typename TestType::DTArg;
    using DTIns = typename TestType::DTIns;

    const DTArg *arg = nullptr;
    const DTIns *ins = nullptr;
    std::pair<size_t, size_t> lowerInclUpperExcl = {-1, -1};
    const DTRes *exp = nullptr;

    SECTION("1x1 matrices") {
        lowerInclUpperExcl = {0, 1};
        SECTION("zero to zero") {
            arg = genGivenVals<DTArg>(1, {0});
            ins = genGivenVals<DTIns>(1, {0});
            exp = genGivenVals<DTRes>(1, {0});
        }
        SECTION("zero to non-zero") {
            arg = genGivenVals<DTArg>(1, {0});
            ins = genGivenVals<DTIns>(1, {2});
            exp = genGivenVals<DTRes>(1, {2});
        }
        SECTION("non-zero to zero") {
            arg = genGivenVals<DTArg>(1, {1});
            ins = genGivenVals<DTIns>(1, {0});
            exp = genGivenVals<DTRes>(1, {0});
        }
        SECTION("non-zero to non-zero") {
            arg = genGivenVals<DTArg>(1, {1});
            ins = genGivenVals<DTIns>(1, {2});
            exp = genGivenVals<DTRes>(1, {2});
        }
    }

    SECTION("nx1 matrices") {
        ins = genGivenVals<DTIns>(3, {10, 11, 12});
        SECTION("arg: all-zero") {
            arg = genGivenVals<DTArg>(6, {0, 0, 0, 0, 0, 0});
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 3};
                exp = genGivenVals<DTRes>(6, {10, 11, 12, 0, 0, 0});
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 4};
                exp = genGivenVals<DTRes>(6, {0, 10, 11, 12, 0, 0});
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {3, 6};
                exp = genGivenVals<DTRes>(6, {0, 0, 0, 10, 11, 12});
            }
        }
        SECTION("arg: almost-all-zero") {
            arg = genGivenVals<DTArg>(6, {0, 2, 0, 4, 0, 0});
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 3};
                exp = genGivenVals<DTRes>(6, {10, 11, 12, 4, 0, 0});
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 4};
                exp = genGivenVals<DTRes>(6, {0, 10, 11, 12, 0, 0});
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {3, 6};
                exp = genGivenVals<DTRes>(6, {0, 2, 0, 10, 11, 12});
            }
        }
        SECTION("arg: almost-all-non-zero") {
            arg = genGivenVals<DTArg>(6, {1, 0, 3, 0, 5, 6});
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 3};
                exp = genGivenVals<DTRes>(6, {10, 11, 12, 0, 5, 6});
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 4};
                exp = genGivenVals<DTRes>(6, {1, 10, 11, 12, 5, 6});
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {3, 6};
                exp = genGivenVals<DTRes>(6, {1, 0, 3, 10, 11, 12});
            }
        }
        SECTION("arg: all-non-zero") {
            arg = genGivenVals<DTArg>(6, {1, 2, 3, 4, 5, 6});
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 3};
                exp = genGivenVals<DTRes>(6, {10, 11, 12, 4, 5, 6});
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 4};
                exp = genGivenVals<DTRes>(6, {1, 10, 11, 12, 5, 6});
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {3, 6};
                exp = genGivenVals<DTRes>(6, {1, 2, 3, 10, 11, 12});
            }
        }
    }

    SECTION("1xm matrices") {
        ins = genGivenVals<DTIns>(1, {11, 12, 13, 14, 15, 16});
        lowerInclUpperExcl = {0, 1};
        SECTION("arg: all-zero") {
            arg = genGivenVals<DTArg>(1, {0, 0, 0, 0, 0, 0});
            exp = genGivenVals<DTRes>(1, {11, 12, 13, 14, 15, 16});
        }
        SECTION("arg: almost-all-zero") {
            arg = genGivenVals<DTArg>(1, {0, 2, 0, 4, 0, 0});
            exp = genGivenVals<DTRes>(1, {11, 12, 13, 14, 15, 16});
        }
        SECTION("arg: almost-all-non-zero") {
            arg = genGivenVals<DTArg>(1, {1, 0, 3, 0, 5, 6});
            exp = genGivenVals<DTRes>(1, {11, 12, 13, 14, 15, 16});
        }
        SECTION("arg: all-non-zero") {
            arg = genGivenVals<DTArg>(1, {1, 2, 3, 4, 5, 6});
            exp = genGivenVals<DTRes>(1, {11, 12, 13, 14, 15, 16});
        }
    }

    SECTION("nxm matrices") {
        ins = genGivenVals<DTIns>(2, {101, 102, 103, 104, 105, 106}); // 2x3
        SECTION("arg: all-zero") {
            arg = genGivenVals<DTArg>(4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}); // 4x3
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 2};
                exp = genGivenVals<DTRes>(4, {101, 102, 103, 104, 105, 106, 0, 0, 0, 0, 0, 0}); // 4x3
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 3};
                exp = genGivenVals<DTRes>(4, {0, 0, 0, 101, 102, 103, 104, 105, 106, 0, 0, 0}); // 4x3
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {2, 4};
                exp = genGivenVals<DTRes>(4, {0, 0, 0, 0, 0, 0, 101, 102, 103, 104, 105, 106}); // 4x3
            }
        }
        SECTION("arg: almost-all-zero") {
            arg = genGivenVals<DTArg>(4, {0, 0, 3, 0, 0, 0, 0, 8, 0, 0, 11, 0}); // 4x3
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 2};
                exp = genGivenVals<DTRes>(4, {101, 102, 103, 104, 105, 106, 0, 8, 0, 0, 11, 0}); // 4x3
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 3};
                exp = genGivenVals<DTRes>(4, {0, 0, 3, 101, 102, 103, 104, 105, 106, 0, 11, 0}); // 4x3
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {2, 4};
                exp = genGivenVals<DTRes>(4, {0, 0, 3, 0, 0, 0, 101, 102, 103, 104, 105, 106}); // 4x3
            }
        }
        SECTION("arg: almost-all-non-zero") {
            arg = genGivenVals<DTArg>(4, {1, 2, 0, 4, 5, 6, 7, 0, 9, 10, 0, 12}); // 4x3
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 2};
                exp = genGivenVals<DTRes>(4, {101, 102, 103, 104, 105, 106, 7, 0, 9, 10, 0, 12}); // 4x3
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 3};
                exp = genGivenVals<DTRes>(4, {1, 2, 0, 101, 102, 103, 104, 105, 106, 10, 0, 12}); // 4x3
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {2, 4};
                exp = genGivenVals<DTRes>(4, {1, 2, 0, 4, 5, 6, 101, 102, 103, 104, 105, 106}); // 4x3
            }
        }
        SECTION("arg: all-non-zero") {
            arg = genGivenVals<DTArg>(4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}); // 4x3
            SECTION("insert at lower bound") {
                lowerInclUpperExcl = {0, 2};
                exp = genGivenVals<DTRes>(4, {101, 102, 103, 104, 105, 106, 7, 8, 9, 10, 11, 12}); // 4x3
            }
            SECTION("insert at middle") {
                lowerInclUpperExcl = {1, 3};
                exp = genGivenVals<DTRes>(4, {1, 2, 3, 101, 102, 103, 104, 105, 106, 10, 11, 12}); // 4x3
            }
            SECTION("insert at upper bound") {
                lowerInclUpperExcl = {2, 4};
                exp = genGivenVals<DTRes>(4, {1, 2, 3, 4, 5, 6, 101, 102, 103, 104, 105, 106}); // 4x3
            }
        }
    }

    checkInsertRow<DTRes, DTArg, DTIns, size_t>(arg, ins, lowerInclUpperExcl.first, lowerInclUpperExcl.second, exp);

    DataObjectFactory::destroy(arg, ins);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertRow - invalid", TAG_KERNELS, (DATA_TYPE_COMBINATIONS), (VALUE_TYPES)) {
    using DTRes = typename TestType::DTRes;
    using DTArg = typename TestType::DTArg;
    using DTIns = typename TestType::DTIns;

    auto arg = genGivenVals<DTArg>(4, {1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12}); // 4x3
    auto ins = genGivenVals<DTIns>(2, {2, -2, 2, 7, 9, 11});                        // 2x3

    size_t lowerIncl;
    size_t upperExcl;

    SECTION("out of bounds (negative)") {
        lowerIncl = -5;
        upperExcl = -3;
    }
    SECTION("out of bounds (too high)") {
        lowerIncl = 3;
        upperExcl = 5;
    }

    checkInsertRowThrow<DTRes>(arg, ins, lowerIncl, upperExcl);

    DataObjectFactory::destroy(arg, ins);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertRow - string specific", TAG_KERNELS, (DATA_TYPES), (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(4, {VT("a"), VT(""), VT("1"), VT("abc"), VT("abc"), VT("abcd"), VT(" "), VT("a"),
                                    VT("ABC"), VT("34ab"), VT("ac"), VT("b")});

    auto ins = genGivenVals<DT>(2, {VT("a"), VT("b"), VT("c"), VT("d"), VT("e"), VT("f")});

    SECTION("multiple insertions, lower bound") {
        size_t lowerIncl = 0;
        size_t upperExcl = 2;
        DT *exp = genGivenVals<DT>(4, {VT("a"), VT("b"), VT("c"), VT("d"), VT("e"), VT("f"), VT(" "), VT("a"),
                                       VT("ABC"), VT("34ab"), VT("ac"), VT("b")});

        checkInsertRow<DT>(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertion, middle") {
        size_t lowerIncl = 1;
        size_t upperExcl = 3;
        DT *exp = genGivenVals<DT>(4, {VT("a"), VT(""), VT("1"), VT("a"), VT("b"), VT("c"), VT("d"), VT("e"), VT("f"),
                                       VT("34ab"), VT("ac"), VT("b")});

        checkInsertRow<DT>(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertions, upper bound") {
        size_t lowerIncl = 2;
        size_t upperExcl = 4;
        DT *exp = genGivenVals<DT>(4, {VT("a"), VT(""), VT("1"), VT("abc"), VT("abc"), VT("abcd"), VT("a"), VT("b"),
                                       VT("c"), VT("d"), VT("e"), VT("f")});

        checkInsertRow<DT>(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("out of bounds - negative") {
        size_t lowerIncl = -5;
        size_t upperExcl = -3;

        checkInsertRowThrow<DT>(arg, ins, lowerIncl, upperExcl);
    }

    SECTION("out of bounds - too high") {
        size_t lowerIncl = 3;
        size_t upperExcl = 5;

        checkInsertRowThrow<DT>(arg, ins, lowerIncl, upperExcl);
    }

    DataObjectFactory::destroy(arg, ins);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertRow - FP specific", TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(4, {
                                       1,
                                       -2,
                                       3,
                                       4,
                                       -5,
                                       6,
                                       7,
                                       -8,
                                       9,
                                       10,
                                       -11,
                                       12.4,
                                   });

    auto ins = genGivenVals<DT>(2, {
                                       2,
                                       -2,
                                       2,
                                       7,
                                       9,
                                       11,
                                   });

    SECTION("multiple insertions, FP bounds") {
        VT lowerIncl = 2.4;
        VT upperExcl = 4.9;
        DT *exp = genGivenVals<DT>(4, {
                                          1,
                                          -2,
                                          3,
                                          4,
                                          -5,
                                          6,
                                          2,
                                          -2,
                                          2,
                                          7,
                                          9,
                                          11,
                                      });

        checkInsertRow<DT>(arg, ins, lowerIncl, upperExcl, exp);
    }

    DataObjectFactory::destroy(arg, ins);
}