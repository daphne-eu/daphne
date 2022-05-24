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
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datagen/GenGivenVals.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE("DenseMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
    // No assertions in this test case. We just want to see if it runs without
    // crashing.
    
    using ValueType = TestType;
    
    const size_t numRows = 10000;
    const size_t numCols = 2000;
    
    DenseMatrix<ValueType> * m = DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, numCols, false);
    
    ValueType * values = m->getValues();
    const size_t numCells = numRows * numCols;
    
    // Fill the matrix with ones of the respective value type.
    for(size_t i = 0; i < numCells; i++)
        values[i] = ValueType(1);
    
    DataObjectFactory::destroy(m);
}

TEST_CASE("DenseMatrix for strings", TAG_KERNELS) {
    
    const size_t numRows = 3;
    const size_t numCols = 4;
    const size_t bytesPerCell = 1;

    SECTION("Append") {
        auto m = DataObjectFactory::create<DenseMatrix<const char*>>(numRows, numCols, false);
        auto exp = genGivenVals<DenseMatrix<const char*>>(numRows, {"0", "", "", "3", 
                                                                    "10", "", "", "13",    
                                                                    "20", "", "", "23"});
        m->prepareAppend();
        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++)
                if(c % 3 == 0)
                    m->append(r, c, std::string(std::to_string(r*10+c)).c_str());

        m->finishAppend();
        CHECK(*m == *exp);
        DataObjectFactory::destroy(m);
        DataObjectFactory::destroy(exp);
    }

    SECTION("Set") {
        auto exp1 = genGivenVals<DenseMatrix<const char*>>(numRows, {"", "1", "", "3",
                                                                    "", "11", "", "13",
                                                                    "", "21", "", "23"});

        auto exp2 = genGivenVals<DenseMatrix<const char*>>(numRows, {"0", "1" ,"2", "3",
                                                                    "10", "11", "12", "13",
                                                                    "20", "21", "22", "23"});
        DenseMatrix<const char*> * m = DataObjectFactory::create<DenseMatrix<const char*>>(numRows, numCols, false, numRows*numCols*bytesPerCell);

        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++){
                size_t num = r*10+c;
                if(num % 2){
                    // m->set(r, c, std::string(num, 'X').c_str());
                    m->set(r, c, std::string(std::to_string(num)).c_str());
                }
            }
        CHECK(*m == *exp1);
        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++){
                size_t num = r*10+c;
                if(!(num % 2)){
                    // m->set(r, c, std::string(num, 'Y').c_str());
                    m->set(r, c, std::string(std::to_string(num)).c_str());
                }
            }
        CHECK(*m == *exp2);
        DataObjectFactory::destroy(m);
        DataObjectFactory::destroy(exp1);
        DataObjectFactory::destroy(exp2);
    }

    SECTION("Append + Set") {

        auto exp1 = genGivenVals<DenseMatrix<const char*>>(numRows, {"0", "", "", "3", 
                                                                    "10", "", "", "13",    
                                                                    "20", "", "", "23"});

        auto exp2 = genGivenVals<DenseMatrix<const char*>>(numRows, {"0", "", "", "3", 
                                                                    "10", std::string(100, 'O').c_str(), "", "13",    
                                                                    "20", "", "", "23"});

        auto exp3 = genGivenVals<DenseMatrix<const char*>>(numRows, {"0", "", "", "3", 
                                                                    "10", std::string(100, 'O').c_str(), "", "13",    
                                                                    "20", std::string(5000, 'X').c_str(), "", "23"});

        auto exp4 = genGivenVals<DenseMatrix<const char*>>(numRows, {"0", "", "", "3", 
                                                                    "10", std::string(100, 'O').c_str(), "", "13",    
                                                                    "20", std::string(5, 'X').c_str(), "", "23"});

        DenseMatrix<const char*> * m = DataObjectFactory::create<DenseMatrix<const char*>>(numRows, numCols, false, numRows*numCols*bytesPerCell);
        m->set(1, 1, std::string(20, 'C').c_str());
        m->prepareAppend();
        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++)
                if(c % 3 == 0)
                    m->append(r, c, std::string(std::to_string(r*10+c)).c_str());

        m->finishAppend();

        CHECK(*m == *exp1);
        m->set(1, 1, std::string(100, 'O').c_str());
        CHECK(*m == *exp2);
        m->set(2, 1, std::string(5000, 'X').c_str());
        CHECK(*m == *exp3);
        m->set(2, 1, std::string(5, 'X').c_str());
        CHECK(*m == *exp4);
        DataObjectFactory::destroy(m);
        DataObjectFactory::destroy(exp1);
        DataObjectFactory::destroy(exp2);
        DataObjectFactory::destroy(exp3);
        DataObjectFactory::destroy(exp4);
    }

    SECTION("View") {
        auto exp1 = genGivenVals<DenseMatrix<const char*>>(2, {"1", "2", "11", "12"});
        auto exp2 = genGivenVals<DenseMatrix<const char*>>(2, {"1", "2", "11", std::string(5, 'X').c_str()});
        auto exp3 = genGivenVals<DenseMatrix<const char*>>(3, {"0", "1", "2", "3",
                                                                "10", "11", std::string(5, 'X').c_str(), "13",
                                                                "20", "21", "22", "23"});
        DenseMatrix<const char*> * m = DataObjectFactory::create<DenseMatrix<const char*>>(numRows, numCols, false, numRows*numCols*bytesPerCell);
        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++)
                m->set(r, c, std::string(std::to_string(r*10+c)).c_str());
        auto mView = DataObjectFactory::create<DenseMatrix<const char*>>(m, 0, 2, 1, 3);

        CHECK(*mView == *exp1);
        
        mView->set(1, 1, std::string(5, 'X').c_str());
        CHECK(*mView == *exp2);
        CHECK(*m == *exp3);

        DataObjectFactory::destroy(m);
        DataObjectFactory::destroy(exp1);
        DataObjectFactory::destroy(exp2);
        DataObjectFactory::destroy(mView);
    }
}

TEST_CASE("DenseMatrix sub-matrix works properly", TAG_DATASTRUCTURES) {
    using ValueType = uint64_t;
    
    const size_t numRowsOrig = 10;
    const size_t numColsOrig = 7;
    const size_t numCellsOrig = numRowsOrig * numColsOrig;
    
    DenseMatrix<ValueType> * mOrig = DataObjectFactory::create<DenseMatrix<ValueType>>(numRowsOrig, numColsOrig, true);
    DenseMatrix<ValueType> * mSub = DataObjectFactory::create<DenseMatrix<ValueType>>(mOrig, 3, 5, 1, 4);
    
    // Sub-matrix dimensions are as expected.
    CHECK(mSub->getNumRows() == 2);
    CHECK(mSub->getNumCols() == 3);
    CHECK(mSub->getRowSkip() == numColsOrig);

    // Sub-matrix shares data array with original.
    ValueType * valuesOrig = mOrig->getValues();
    ValueType * valuesSub = mSub->getValues();
    CHECK((valuesSub >= valuesOrig && valuesSub < valuesOrig + numCellsOrig));
    valuesSub[0] = 123;
    CHECK(valuesOrig[3 * numColsOrig + 1] == 123);
    
    // Freeing both matrices does not result in double-free errors.
    SECTION("Freeing the original matrix first is fine") {
        DataObjectFactory::destroy(mOrig);
        DataObjectFactory::destroy(mSub);
    }
    SECTION("Freeing the sub-matrix first is fine") {
        DataObjectFactory::destroy(mSub);
        DataObjectFactory::destroy(mOrig);
    }
}