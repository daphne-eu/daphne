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
#include <runtime/local/kernels/Sample.h>

#include <tags.h>

#include <catch.hpp>

TEMPLATE_PRODUCT_TEST_CASE("Sample", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT * m = nullptr;    
    const size_t size = 10000;
    const VT range = 10000;
    bool withReplacement;    
    
    SECTION("with replacement"){
        withReplacement = true;
    
        sample<DT, VT>(m, range, size, withReplacement, -1, nullptr);

        REQUIRE(m->getNumRows() == size);
        REQUIRE(m->getNumCols() == 1);

        VT *values = m->getValues();        
        for (size_t i=0; i<size; i++){                
            CHECK(values[i] >= 0);            
            CHECK(values[i] < range);            
        }
    }
        
    SECTION("without replacement"){
        withReplacement = false;
    
        sample<DT, VT>(m, range, size, withReplacement, -1, nullptr);
        
        REQUIRE(m->getNumRows() == size);
        REQUIRE(m->getNumCols() == 1);
        
        VT *values = m->getValues();        
        
        std::sort(values, values + size);   
        CHECK(values[0] >= 0);                   
        CHECK(values[size-1] < range);                   
        for (size_t i=1; i<size; i++){                
            CHECK(values[i-1] != values[i]);            
        }
    }

    DataObjectFactory::destroy(m);
}