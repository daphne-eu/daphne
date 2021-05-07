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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/RandMatrix.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("RandMatrix, full/empty", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    const size_t numRows = 3;
    const size_t numCols = 4;
    const VT min = 100;
    const VT max = 200;
    
    DT * m = nullptr;
    
    SECTION("full") {
        randMatrix<DT, VT>(m, numRows, numCols, min, max, 1.0, -1);

        REQUIRE(m->getNumRows() == numRows);
        REQUIRE(m->getNumCols() == numCols);

        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++) {
                CHECK(m->get(r, c) >= min);
                CHECK(m->get(r, c) <= max);
            }
    }
    SECTION("empty") {
        randMatrix<DT, VT>(m, numRows, numCols, min, max, 0.0, -1);

        REQUIRE(m->getNumRows() == numRows);
        REQUIRE(m->getNumCols() == numCols);

        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++) {
                CHECK(m->get(r, c) == 0);
            }
    }
    
    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("RandMatrix, sparse", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    const size_t numRows = 1000;
    const size_t numCols = 1000;
    const VT min = 100;
    const VT max = 200;
    const double sparsity = 0.9;
    
    DT * m = nullptr;
    randMatrix<DT, VT>(m, numRows, numCols, min, max, sparsity, -1);

    size_t numNonZeros = 0;
    for(size_t r = 0; r < numRows; r++)
        for(size_t c = 0; c < numCols; c++)
            if(m->get(r, c) != 0)
                numNonZeros++;
    
    const double sparsityFound = static_cast<double>(numNonZeros) / (numRows * numCols);
    CHECK(abs(sparsityFound - sparsity) < 0.001);
        
    DataObjectFactory::destroy(m);
}