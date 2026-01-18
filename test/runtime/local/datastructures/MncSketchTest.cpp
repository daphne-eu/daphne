/*
 * Tests for MNC sketch on CSRMatrix
 */

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <runtime/local/datastructures/MncSketch.h>

#include <tags.h>
#include <catch.hpp>

#include <cstdint>
#include <vector>

// run ./test.sh -nb -d yes [datastructures] after building Daphne to execute this test
TEST_CASE("MNC sketch from CSRMatrix basic", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix:
    // [0 5 0
    //  0 0 3
    //  1 0 0]
    const size_t numRows     = 3;
    const size_t numCols     = 3;
    const size_t numNonZeros = 3;

    CSRMatrix<ValueType> *m =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, /*zero=*/true);

    ValueType *values   = m->getValues();
    size_t   *colIdxs   = m->getColIdxs();
    size_t   *rowOffsets= m->getRowOffsets();

    // Valid CSR: one nnz per row
    // rowOffsets: [0,1,2,3]
    rowOffsets[0] = 0;
    rowOffsets[1] = 1;
    rowOffsets[2] = 2;
    rowOffsets[3] = 3;

    // colIdxs: (0,1)->5, (1,2)->3, (2,0)->1
    colIdxs[0] = 1;
    colIdxs[1] = 2;
    colIdxs[2] = 0;

    values[0] = 5.0;
    values[1] = 3.0;
    values[2] = 1.0;

    MncSketch h = buildMncFromCsr(*m);

    // dimensions
    CHECK(h.m == numRows);
    CHECK(h.n == numCols);

    // row + col nnz
    std::vector<std::uint32_t> expectedHr{1,1,1};
    std::vector<std::uint32_t> expectedHc{1,1,1};
    CHECK(h.hr == expectedHr);
    CHECK(h.hc == expectedHc);

    // summary stats
    CHECK(h.maxHr    == 1);
    CHECK(h.maxHc    == 1);
    CHECK(h.nnzRows  == 3);
    CHECK(h.nnzCols  == 3);
    CHECK(h.rowsEq1  == 3);
    CHECK(h.colsEq1  == 3);
    CHECK(h.rowsGtHalf == 0); // 1 <= n/2 since n=3
    CHECK(h.colsGtHalf == 0); // 1 <= m/2 since m=3

    DataObjectFactory::destroy(m);
}

TEST_CASE("MNC sketch respects CSRMatrix sub-matrix view", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Original 4x3 matrix:
    // row 0: (0,1)
    // row 1: (1,2)
    // row 2: (2,0)
    // row 3: (3,2)  <- we'll slice rows [1,3) = rows 1 and 2
    const size_t numRowsOrig   = 4;
    const size_t numColsOrig   = 3;
    const size_t numNonZeros   = 4;

    CSRMatrix<ValueType> *mOrig =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsOrig, numColsOrig, numNonZeros, /*zero=*/true);

    ValueType *valuesOrig    = mOrig->getValues();
    size_t   *colIdxsOrig    = mOrig->getColIdxs();
    size_t   *rowOffsetsOrig = mOrig->getRowOffsets();

    rowOffsetsOrig[0] = 0;
    rowOffsetsOrig[1] = 1;
    rowOffsetsOrig[2] = 2;
    rowOffsetsOrig[3] = 3;
    rowOffsetsOrig[4] = 4;

    colIdxsOrig[0] = 1; // row 0
    colIdxsOrig[1] = 2; // row 1
    colIdxsOrig[2] = 0; // row 2
    colIdxsOrig[3] = 2; // row 3

    valuesOrig[0] = 1.0;
    valuesOrig[1] = 1.0;
    valuesOrig[2] = 1.0;
    valuesOrig[3] = 1.0;

    // Create sub-matrix with rows [1,3) = rows 1 and 2 of original
    CSRMatrix<ValueType> *mSub = DataObjectFactory::create<CSRMatrix<ValueType>>(mOrig, 1, 3);

    MncSketch hSub = buildMncFromCsr(*mSub);

    // submatrix is 2x3, with nnz rows = 2
    CHECK(hSub.m == 2);
    CHECK(hSub.n == 3);

    // each row in the submatrix has exactly 1 nnz
    std::vector<std::uint32_t> expectedHrSub{1,1};
    CHECK(hSub.hr == expectedHrSub);
    CHECK(hSub.nnzRows == 2);
    CHECK(hSub.rowsEq1 == 2);

    DataObjectFactory::destroy(mSub);
    DataObjectFactory::destroy(mOrig);
}

TEST_CASE("MNC Sketch example from paper", TAG_DATASTRUCTURES) {
    using ValueType = double;
    /* Matrix:
    [0,0,0,0,0,0,0,1,0], 
    [0,1,0,0,1,0,0,0,0], 
    [0,0,0,1,1,1,0,0,0], 
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [1,0,0,1,0,0,0,0,0],
    [0,0,1,0,1,0,1,0,0],
    [0,0,0,0,0,0,0,0,1],
    */
    const size_t numRows     = 9;
    const size_t numCols     = 9;
    const size_t numNonZeros = 14;

    CSRMatrix<ValueType> *m =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, /*zero=*/true);

    ValueType *values   = m->getValues();
    size_t   *colIdxs   = m->getColIdxs();
    size_t   *rowOffsets= m->getRowOffsets();

    // Valid CSR: one nnz per row
    // rowOffsets: [ 0  1  3  6  6  7  8 10 13 14]
    rowOffsets[0] = 0;
    rowOffsets[1] = 1;
    rowOffsets[2] = 3;
    rowOffsets[3] = 6;
    rowOffsets[4] = 6;
    rowOffsets[5] = 7;
    rowOffsets[6] = 8;
    rowOffsets[7] = 10;
    rowOffsets[8] = 13;
    rowOffsets[9] = 14;

    // colIdxs: [7 1 4 3 4 5 2 7 0 3 2 4 6 8]
    colIdxs[0] = 7;
    colIdxs[1] = 1;
    colIdxs[2] = 4;
    colIdxs[3] = 3;
    colIdxs[4] = 4;
    colIdxs[5] = 5;
    colIdxs[6] = 2;
    colIdxs[7] = 7;
    colIdxs[8] = 0;
    colIdxs[9] = 3;
    colIdxs[10] = 2;
    colIdxs[11] = 4;
    colIdxs[12] = 6;
    colIdxs[13] = 8;
    
    for (size_t i = 0; i < numNonZeros; ++i)
        values[i] = 1.0;

    MncSketch h = buildMncFromCsr(*m);

    // dimensions
    CHECK(h.m == numRows);
    CHECK(h.n == numCols);

    // row + col nnz
    
    std::vector<std::uint32_t> expectedHr{1,2,3,0,1,1,2,3,1};
    std::vector<std::uint32_t> expectedHc{1,1,2,2,3,1,1,2,1};
    CHECK(h.hr == expectedHr);
    CHECK(h.hc == expectedHc);

    // her and hec
    std::vector<std::uint32_t> expectedHer{0,1,1,0,0,0,1,1,1};
    std::vector<std::uint32_t> expectedHec{0,0,1,0,0,0,0,2,1};
    std::vector<std::uint32_t> notexpectedHec{1,1,1,0,0,0,0,2,1};
    CHECK(h.her == expectedHer);
    CHECK(h.hec == expectedHec);

    DataObjectFactory::destroy(m);
}

// Tests for estimateSparsity_product function
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

    double s = estimateSparsity_product(hA, hB);

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

    double s = estimateSparsity_product(hA, hB);

    REQUIRE(s >= 0.0);
    REQUIRE(s <= 1.0);
    // REQUIRE(s == 5);
    // std::size_t p = (hA.nnzRows - hA.rowsEq1) * (hB.nnzCols - hB.colsEq1);
    // REQUIRE(p == 4);           
    // REQUIRE(s == 5);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}
