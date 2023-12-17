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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/InsertRow.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define VALUE_TYPES int32_t, double

TEMPLATE_TEST_CASE("InsertRow - dense matrix", TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;

    auto arg = genGivenVals<DenseMatrix<VT>>(3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    });

    auto ins = genGivenVals<DenseMatrix<VT>>(2, {
        0, 2, 1,
        1, 1, 1,
    });

    size_t lowerIncl = 0;
    size_t upperExcl = 2;

    std::cout << "before test" << std::endl;

    SECTION("test func") {
        /*
        auto dense_exp = genGivenVals<DenseMatrix<VT>>({
        });
        */
        std::cout << *arg << std::endl;
        std::cout << *ins << std::endl;
        std::cout << lowerIncl << std::endl;
        std::cout << upperExcl << std::endl;
        std::cout << "end of section" << std::endl;
    }
    std::cout << "after section" << std::endl;


    DenseMatrix<VT> * res = nullptr;

    

    insertRow<DenseMatrix<VT>, DenseMatrix<VT>>(res, arg, ins, lowerIncl, upperExcl, nullptr);

    std::cout << *res << std::endl;
    std::cout << "end" << std::endl;

    DataObjectFactory::destroy(arg, ins, res);
}