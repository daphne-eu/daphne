#pragma once

#include <iostream>

#include <vector>
#include <cstddef>
#include <cstdint>

#include <runtime/local/datastructures/CSRMatrix.h>

class MncSketch{
  public:
    // dimensions
    std::size_t m = 0; // rows
    std::size_t n = 0; // cols

    // core counts
    std::vector<std::uint32_t> hr;   // nnz per row (size m)
    std::vector<std::uint32_t> hc;   // nnz per col (size n)

    // Extended counts (optional; can be empty)
    std::vector<std::uint32_t> her;  // nnz in row i that lie in columns with hc == 1
    std::vector<std::uint32_t> hec;  // nnz in column j that lie in rows with hr == 1

    // Summary statistics
    std::uint32_t maxHr = 0;
    std::uint32_t maxHc = 0;
    std::uint32_t nnzRows = 0;       // # rows with hr > 0
    std::uint32_t nnzCols = 0;       // # cols with hc > 0
    std::uint32_t rowsEq1 = 0;       // # rows with hr == 1
    std::uint32_t colsEq1 = 0;       // # cols with hc == 1
    std::uint32_t rowsGtHalf = 0;    // # rows with hr > n/2
    std::uint32_t colsGtHalf = 0;    // # cols with hc > m/2
    bool isDiagonal = false;         // optional flag if A is (full) diagonal

  
    size_t getNumRows() const { return m; }
    size_t getNumCols() const { return n; }
    const std::vector<std::uint32_t>& getHr() const { return hr; }
    const std::vector<std::uint32_t>& getHc() const { return hc; }
    std::uint32_t getMaxHr() const { return maxHr; }
    std::uint32_t getMaxHc() const { return maxHc; }
    std::uint32_t getNnzRows() const { return nnzRows; }
    std::uint32_t getNnzCols() const { return nnzCols; }
    std::uint32_t getRowsEq1() const { return rowsEq1; }
    std::uint32_t getColsEq1() const { return colsEq1; }
    std::uint32_t getRowsGtHalf() const { return rowsGtHalf; }
    std::uint32_t getColsGtHalf() const { return colsGtHalf; }
    bool getIsDiagonal() const { return isDiagonal; }    
};

// Build MNC sketch from a DAPHNE CSRMatrix
template<typename VT>
MncSketch buildMncFromCsr(const CSRMatrix<VT> &A) {
    MncSketch h;
    h.m = A.getNumRows();
    h.n = A.getNumCols();

    h.hr.assign(h.m, 0);
    h.hc.assign(h.n, 0);

    const std::size_t *rowOffsets = A.getRowOffsets();
    const std::size_t *colIdxs    = A.getColIdxs();

    // --- 1) per-row nnz (hr) and row stats ---
    for(std::size_t i = 0; i < h.m; ++i) {
        // row i uses indices [rowOffsets[i], rowOffsets[i+1])
        std::size_t s   = rowOffsets[i];
        std::size_t e   = rowOffsets[i+1];
        std::uint32_t cnt = static_cast<std::uint32_t>(e - s);

        h.hr[i] = cnt;

        if(cnt > 0) {
            h.nnzRows++;
            if(cnt == 1)
                h.rowsEq1++;
            if(cnt > h.n / 2)
                h.rowsGtHalf++;
        }
        if(cnt > h.maxHr)
            h.maxHr = cnt;
    }

    // --- 2) per-column nnz (hc) and column stats ---
    // We must iterate all nnz in this *view*:
    std::size_t nnzBegin = rowOffsets[0];
    std::size_t nnzEnd   = rowOffsets[h.m];

    for(std::size_t k = nnzBegin; k < nnzEnd; ++k) {
        std::size_t j = colIdxs[k];
        auto &cnt = h.hc[j];
        if(cnt == 0)
            h.nnzCols++;
        cnt++;
    }

    for(std::size_t j = 0; j < h.n; ++j) {
        auto cnt = h.hc[j];
        if(cnt == 1)
            h.colsEq1++;
        if(cnt > h.m / 2)
            h.colsGtHalf++;
        if(cnt > h.maxHc)
            h.maxHc = cnt;
    }

    return h;
}
