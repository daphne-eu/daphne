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



#define VALUE_TYPES double
#define DATA_TYPES CSRMatrix, DenseMatrix

template<class DT, typename VT>
void checkReplace(DT* outputMatrix, DT* inputMatrix,VT pattern, VT replacement, const DT* expected){
	replace<DT, DT, VT>(outputMatrix, inputMatrix, pattern, replacement);
	CHECK(*outputMatrix == *expected);
}

TEMPLATE_PRODUCT_TEST_CASE("Replace", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){
	using DT = TestType;

	//inplace updates

	auto initMatrix = genGivenVals<DT>(4, {
	1, 2, 3, 7, 7, 7,
	7, 1, 2, 3, 7, 7,
	7, 7, 1, 2, 3, 7,
	7, 7, 7, 1, 2, 3,
	});

	auto testMatrix1 = genGivenVals<DT>(4, {
        7, 2, 3, 7, 7, 7,
        7, 7, 2, 3, 7, 7,
        7, 7, 7, 2, 3, 7,
        7, 7, 7, 7, 2, 3,
	});

	auto testMatrix2 = genGivenVals<DT>(4, {
        7, 7, 3, 7, 7, 7,
        7, 7, 7, 3, 7, 7,
        7, 7, 7, 7, 3, 7,
        7, 7, 7, 7, 7, 3,
	});
	double target=1;
	double replacement=7;
	checkReplace(initMatrix, initMatrix ,target, replacement, testMatrix1);
        //should do nothing because there is no ones
	checkReplace(initMatrix, initMatrix, target, replacement, testMatrix1);
	target=2;
	checkReplace(initMatrix, initMatrix, target, replacement, testMatrix2);


 	// update in a new copy
 	auto testMatrix3 = genGivenVals<DT>(4, {
	 7, 7, 7, 7, 7, 7,
	 7, 7, 7, 7, 7, 7,
	 7, 7, 7, 7, 7, 7,
	 7, 7, 7, 7, 7, 7,
 	});

 	auto testMatrix4 = genGivenVals<DT>(4, {
	 7, 7, 10, 7, 7, 7,
	 7, 7, 7, 10, 7, 7,
	 7, 7, 7, 7, 10, 7,
	 7, 7, 7, 7, 7, 10,
	 });

	DT * outputMatrix=nullptr;
	target=3;
	checkReplace(outputMatrix, initMatrix, target, replacement, testMatrix3);
	replacement=10;
	
	checkReplace(initMatrix, initMatrix, target, replacement, testMatrix4);
	//this test case should act as a copy	
	DT * outputMatrix2=nullptr;
	target=3;
	replacement=3;
	checkReplace(outputMatrix2, initMatrix,  target, replacement, testMatrix4);
	DataObjectFactory::destroy(initMatrix);
  	DataObjectFactory::destroy(testMatrix1);
  	DataObjectFactory::destroy(testMatrix2);
	DataObjectFactory::destroy(testMatrix3);
  	DataObjectFactory::destroy(testMatrix4);
	DataObjectFactory::destroy(outputMatrix);
	DataObjectFactory::destroy(outputMatrix2);

}
