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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/RandMatrix.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>

#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES double, float, uint32_t, uint8_t

TEMPLATE_PRODUCT_TEST_CASE("RandMatrix", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    const size_t numRows = 100;
    const size_t numCols = 50;
    const VT min = 100;
    const VT max = 200;

    for(double sparsity : {0.0, 0.1, 0.5, 0.9, 1.0}) {
        DYNAMIC_SECTION("sparsity = " << sparsity) {
            DT * m = nullptr;
            randMatrix<DT, VT>(m, numRows, numCols, min, max, sparsity, -1, nullptr);

            REQUIRE(m->getNumRows() == numRows);
            REQUIRE(m->getNumCols() == numCols);

            size_t numNonZeros = 0;
            for(size_t r = 0; r < numRows; r++)
                for(size_t c = 0; c < numCols; c++) {
                    const VT v = m->get(r, c);
                    if(v) {
                        CHECK(v >= min);
                        CHECK(v <= max);
                        numNonZeros++;
                    }
                }

            const size_t numNonZerosExpected = size_t(round(sparsity * numRows * numCols));
            CHECK(numNonZerosExpected == numNonZeros);

            DataObjectFactory::destroy(m);
        }
    }
}

