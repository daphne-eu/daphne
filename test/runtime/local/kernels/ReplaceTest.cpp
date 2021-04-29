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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/kernels/Replace.h>
#include <runtime/local/kernels/CheckEq.h>


#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>



#define VALUE_TYPES double, uint32_t
#define DATA_TYPES DenseMatrix, CSRMatrix

template<class DT, typename VT>
void checkReplace(DT* inoutMatrix, VT pattern, VT replacement, const DT* expected){
	replace<DT,VT>(inoutMatrix, pattern, replacement);
	CHECK(*inoutMatrix == *expected);   
}

TEMPLATE_PRODUCT_TEST_CASE("Replace", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){
	using DT = TestType;
	
	auto initMatrix = genGivenVals(4, {
	1, 2, 3, 7, 7, 7,
	7, 1, 2, 3, 7, 7,
	7, 7, 1, 2, 3, 7,
	7, 7, 7, 1, 2, 3,
	});
	
	auto testMatrix1 = genGivenVals(4, {
        7, 2, 3, 7, 7, 7,
        7, 7, 2, 3, 7, 7,
        7, 7, 7, 2, 3, 7,
        7, 7, 7, 7, 2, 3,
	}); 
	
	auto testMatrix2 = genGivenVals(4, {
        7, 7, 3, 7, 7, 7,
        7, 7, 7, 3, 7, 7,
        7, 7, 7, 7, 3, 7,
        7, 7, 7, 7, 7, 3,
	});

	checkReplace(initMatrix, 1, 7, testMatrix1);
        //should do nothing because there is no ones
        checkReplace(initMatrix, 1, 7, testMatrix1);
	checkReplace(initMatrix, 2, 7, testMatrix2);
	
	DataObjectFactory::destroy(initMatrix);
    	DataObjectFactory::destroy(testMatrix1);
    	DataObjectFactory::destroy(testMatrix2);
}

