#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <runtime/local/datastructures/MncSketch.h>
#include <runtime/local/datastructures/MNCSparsityEstimation.h>


#include <tags.h>
#include <catch.hpp>

#include <cstdint>
#include <vector>


// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST_CASE("Case 1: maxHr(A) <= 1 or maxHr(B) <= 1", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix A: 3x3 
    // [1 0 0]
    // [0 1 0]
    // [0 0 1]
    const size_t numRowsA = 3;
    const size_t numColsA = 3;
    const size_t nnzA     = 3;

    CSRMatrix<ValueType> *A = 
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, nnzA, /*zero=*/true);

    ValueType *valuesA    = A->getValues();
    size_t   *colIdxsA    = A->getColIdxs();
    size_t   *rowOffsetsA = A->getRowOffsets();

    rowOffsetsA[0] = 0;
    rowOffsetsA[1] = 1;
    rowOffsetsA[2] = 2;
    rowOffsetsA[3] = 3;

    colIdxsA[0] = 0;
    colIdxsA[1] = 1; 
    colIdxsA[2] = 2; 

    valuesA[0] = 1.0;
    valuesA[1] = 1.0;
    valuesA[2] = 1.0;
    // Matrix B: 3x2 
    // [1 0]
    // [1 0]
    // [0 1]
    const size_t numRowsB = 3;
    const size_t numColsB = 2;
    const size_t nnzB     = 3;

    CSRMatrix<ValueType> *B = 
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, nnzB, /*zero=*/true);

    ValueType *valuesB    = B->getValues();
    size_t   *colIdxsB    = B->getColIdxs();
    size_t   *rowOffsetsB = B->getRowOffsets();

    rowOffsetsB[0] = 0;
    rowOffsetsB[1] = 1;
    rowOffsetsB[2] = 2;
    rowOffsetsB[3] = 3;

    colIdxsB[0] = 0; 
    colIdxsB[1] = 0; 
    colIdxsB[2] = 1; 

    valuesB[0] = 1.0;
    valuesB[1] = 1.0;
    valuesB[2] = 1.0;

    MncSketch hA = buildMncFromCsr(*A);
    MncSketch hB = buildMncFromCsr(*B);

    double s = estimateSparsity(hA, hB);

    // REQUIRE(s >= 0.0);
    // REQUIRE(s <= 1.0);
    REQUIRE(s == 0.5);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}

TEST_CASE("Case 2: some rows/cols have >1 nnz", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // --- Matrix A: 3x3 ---
    // [1 1 0]
    // [0 1 1]
    // [0 0 1]
    const size_t numRowsA = 3;
    const size_t numColsA = 3;
    const size_t nnzA     = 5;
    CSRMatrix<ValueType> *A = 
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, nnzA, true);

    size_t *rowOffsetsA = A->getRowOffsets();
    size_t *colIdxsA    = A->getColIdxs();
    ValueType *valuesA  = A->getValues();

    rowOffsetsA[0] = 0;
    rowOffsetsA[1] = 2; 
    rowOffsetsA[2] = 4; 
    rowOffsetsA[3] = 5; 

    colIdxsA[0] = 0;
    colIdxsA[1] = 1;
    colIdxsA[2] = 1; 
    colIdxsA[3] = 2;
    colIdxsA[4] = 2; 

    for(size_t i = 0; i < nnzA; i++) valuesA[i] = 1.0;

    // --- Matrix B: 3x3 ---
    // [1 0 0]
    // [1 1 0]
    // [0 1 1]
    const size_t numRowsB = 3;
    const size_t numColsB = 3;
    const size_t nnzB     = 5;
    CSRMatrix<ValueType> *B = 
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, nnzB, true);

    size_t *rowOffsetsB = B->getRowOffsets();
    size_t *colIdxsB    = B->getColIdxs();
    ValueType *valuesB  = B->getValues();

    rowOffsetsB[0] = 0;
    rowOffsetsB[1] = 1;
    rowOffsetsB[2] = 3;
    rowOffsetsB[3] = 5;

    colIdxsB[0] = 0; 
    colIdxsB[1] = 0;
    colIdxsB[2] = 1; 
    colIdxsB[3] = 1;
    colIdxsB[4] = 2; 

    for(size_t i = 0; i < nnzB; i++) valuesB[i] = 1.0;

    MncSketch hA = buildMncFromCsr(*A);
    MncSketch hB = buildMncFromCsr(*B);

    double s = estimateSparsity(hA, hB);

    REQUIRE(s >= 0.0);
    REQUIRE(s <= 1.0);
    // REQUIRE(s == 5);
    // std::size_t p = (hA.nnzRows - hA.rowsEq1) * (hB.nnzCols - hB.colsEq1);
    // REQUIRE(p == 4);           
    // REQUIRE(s == 5);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}

