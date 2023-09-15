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

#include "runtime/local/kernels/InPlaceUtils.h"
#include "runtime/local/datastructures/DataObjectFactory.h"
#include <memory>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <optional>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>
#include <tuple>

// ****************************************************************************
// InPlaceUtils::isInPlaceable 
// InPlaceUtils::selectInPlaceableOperand
// ****************************************************************************

// Already implicitly tested in checkSelectInPlaceableOperand
/*
template<class DT>
void checkIsInPlaceable(bool exp, const DT* arg, bool hasFutureUseArg) {
    auto res = InPlaceUtils::isInPlaceable(arg, hasFutureUseArg);
    CHECK(res == exp);
}
*/

template<class DT, typename... Args>
void checkSelectInPlaceableOperand(DT* exp, DT* arg, bool hasFutureUseArg, Args... args) {
    auto res = InPlaceUtils::selectInPlaceableOperand(arg, hasFutureUseArg, args...);
    CHECK(res == exp);
}

TEMPLATE_PRODUCT_TEST_CASE("InPlaceUtils::selectInPlaceableOperand", TAG_INPLACE, (DenseMatrix), (uint32_t)) {
    using DT = TestType;

    DT * lhs = nullptr;
    DT * rhs = nullptr;

    DT * exp_nullptr = nullptr;

    lhs = genGivenVals<DT>(3, {
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
    });

    rhs = genGivenVals<DT>(3, {
            1, 5,  9,
            2, 6, 10,
            3, 7, 11,
    });

    auto variations = {std::make_tuple(true, true, exp_nullptr),
                        std::make_tuple(false, true, lhs),
                        std::make_tuple(true, false, rhs),
                        std::make_tuple(false, false, lhs)};

    //hasFutureUse
    for (const auto &var : variations) {
        DYNAMIC_SECTION("hasFutureUse: " << std::get<0>(var) << " " << std::get<1>(var)) {
            checkSelectInPlaceableOperand(std::get<2>(var), lhs, std::get<0>(var), rhs, std::get<1>(var));
        }
    }

    //getRefCounter
    for (const auto &var : variations) {
        DYNAMIC_SECTION("getRefCounter: " << std::get<0>(var) << " " << std::get<1>(var)) {

            if (std::get<0>(var)) {
                lhs->increaseRefCounter();
            }
            if (std::get<1>(var)) {
                rhs->increaseRefCounter();
            }

            checkSelectInPlaceableOperand(std::get<2>(var), lhs, false, rhs, false);
        }
    }

    //getValuesUseCount
    for (const auto &var : variations) {
        DYNAMIC_SECTION("getValuesUseCount: " << std::get<0>(var) << " " << std::get<1>(var)) {

            DT* view_lhs = nullptr;
            DT* view_rhs = nullptr;

            if (std::get<0>(var)) {
                view_lhs = DataObjectFactory::create<DT>(lhs, 0, 3, 0, 3);
            }
            if (std::get<1>(var)) {
                view_rhs = DataObjectFactory::create<DT>(rhs, 0, 3, 0, 3);
            }

            checkSelectInPlaceableOperand(std::get<2>(var), lhs, false, rhs, false);

            if (std::get<0>(var)) {
                DataObjectFactory::destroy(view_lhs);
            }
            if (std::get<1>(var)) {
                DataObjectFactory::destroy(view_rhs);
            }
        }
    }

    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

// ****************************************************************************
// InPlaceUtils::isValidType?Weak
// ****************************************************************************

template<class DTArg1, typename DTArg2>
void checkIsValidType(bool exp, const DTArg1* arg1, const DTArg2* arg2) {
    auto res = InPlaceUtils::isValidType(arg1, arg2);
    CHECK(res == exp);
}

template<class DTArg1, typename DTArg2>
void checkIsValidTypeWeak(bool exp, const DTArg1* arg1, const DTArg2* arg2) {
    auto res = InPlaceUtils::isValidTypeWeak(arg1, arg2);
    CHECK(res == exp);
}


TEMPLATE_PRODUCT_TEST_CASE("InPlaceUtils::isValidType?Weak", TAG_INPLACE, (DenseMatrix), (uint32_t)) {
    using DT = TestType;

    DT * arg1 = nullptr;
    DT * arg2 = nullptr;
    DT * arg3 = nullptr;
    DT * arg4 = nullptr;
    DenseMatrix<double> * arg5 = nullptr;

    arg1 = genGivenVals<DT>(3, {
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
    });
    arg2 = genGivenVals<DT>(3, {
            1, 5,  9,
            2, 6, 10,
            3, 7, 11,
    });
    arg3 = genGivenVals<DT>(3, {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12,
    });
    arg4 = genGivenVals<DT>(4, {
            1, 2, 3,
            6, 7, 8,
           11,12,13,
           21,22,23
    });
    arg5 = genGivenVals<DenseMatrix<double>>(3, {
            1, 2, 3,
            5, 6, 7,
            9,10,11
    });

    //hasFutureUse
    SECTION("isValidType") {
        checkIsValidType(true, arg1, arg2);
        checkIsValidType(false, arg1, arg3);
        checkIsValidType(false, arg1, arg5);
    }

    SECTION("isValidTypeWeak") {
        checkIsValidTypeWeak(true, arg1, arg2);
        checkIsValidTypeWeak(true, arg3, arg4);
        checkIsValidTypeWeak(false, arg1, arg3);
        checkIsValidTypeWeak(false, arg1, arg5);
    }

    DataObjectFactory::destroy(arg1);
    DataObjectFactory::destroy(arg2);
    DataObjectFactory::destroy(arg3);
    DataObjectFactory::destroy(arg4);
    DataObjectFactory::destroy(arg5);
}

