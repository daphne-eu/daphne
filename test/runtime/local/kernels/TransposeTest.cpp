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

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/Transpose.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

template<class DT>
void checkTranspose(const DT * arg, const DT * exp) {
    DT * res = nullptr;
    transpose<DT, DT>(res, arg, nullptr);
    if constexpr(std::is_same_v<DT, DenseMatrix<const char*>>){
        for(size_t val = 0; val < exp->getNumItems(); val++){
            CHECK(strcmp(res->getValues()[val], exp->getValues()[val]) == 0);
            if(val < exp->getNumItems()-1)
                CHECK((res->getValues()[val] + strlen(res->getValues()[val]) + 1) == res->getValues()[val+1]);
        }
    }
    else
        CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("Transpose", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    DT * m = nullptr;
    DT * mt = nullptr;
    
    SECTION("fully populated matrix") {
        m = genGivenVals<DT>(3, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
        });
        mt = genGivenVals<DT>(4, {
            1, 5,  9,
            2, 6, 10,
            3, 7, 11,
            4, 8, 12,
        });
        checkTranspose(m, mt);
        DataObjectFactory::destroy(m, mt);
    }
    SECTION("sparse matrix") {
        m = genGivenVals<DT>(5, {
            0, 0, 0, 0, 0, 0,
            0, 0, 3, 0, 0, 0,
            0, 0, 0, 0, 4, 0,
            0, 0, 0, 0, 0, 0,
            5, 0, 0, 0, 6, 0,
        });
        mt = genGivenVals<DT>(6, {
            0, 0, 0, 0, 5,
            0, 0, 0, 0, 0,
            0, 3, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 4, 0, 6,
            0, 0, 0, 0, 0,
        });
        checkTranspose(m, mt);
        DataObjectFactory::destroy(m, mt);
    }
    SECTION("empty matrix") {
        m = genGivenVals<DT>(3, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        });
        mt = genGivenVals<DT>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
        checkTranspose(m, mt);
        DataObjectFactory::destroy(m, mt);
    }
    SECTION("String matrix 1") {
        auto mString = genGivenVals<DenseMatrix<const char*>>(3, {
            "1", "2", "3", "4",
            "5", "6", "7", "8",
            "9", "10", "11", "12",
        });
        auto mtString = genGivenVals<DenseMatrix<const char*>>(4, {
            "1", "5",  "9",
            "2", "6", "10",
            "3", "7", "11",
            "4", "8", "12",
        });
        checkTranspose(mString, mtString);
        DataObjectFactory::destroy(mString, mtString);
    }

    SECTION("String matrix 2") {
        auto mString = genGivenVals<DenseMatrix<const char*>>(3, {
            "1", "", "", "",
            "", "6", "", "8",
            "9", "", "", "",
        });
        auto mtString = genGivenVals<DenseMatrix<const char*>>(4, {
            "1", "",  "9",
            "", "6", "",
            "", "", "",
            "", "8", "",
        });
        checkTranspose(mString, mtString);
        DataObjectFactory::destroy(mString, mtString);
    }

    SECTION("String vector 1") {
        auto mString = genGivenVals<DenseMatrix<const char*>>(1, {
            "1", "2", "", "4",
        });
        auto mtString = genGivenVals<DenseMatrix<const char*>>(4, {
            "1",
            "2", 
            "", 
            "4", 
        });
        checkTranspose(mString, mtString);
        checkTranspose(mtString, mString);
        DataObjectFactory::destroy(mString, mtString);
    }
}