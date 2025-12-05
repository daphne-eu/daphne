#include <runtime/local/datastructures/CSRMatrix.h>
#include <iostream>
#include <vector>
#include <cstddef> // std name space

struct MncSketch{
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
    std::uint32_t n_rowsHalfFull = 0;    // # rows with hr > n/2
    std::uint32_t n_colsHalfFull = 0;    // # cols with hc > m/2
    bool isDiagonal = false;         // optional flag if A is (full) diagonal
};

template<class VT>
MncSketch buildMncFromCsr(const CSRMatrix<VT> &A) {
    MncSketch h;
    h.m = A.getNumRows();
    h.n = A.getNumCols();

    h.hr.assign(h.m, 0);
    h.hc.assign(h.n, 0);

    // --- 1) per-row nnz (hr) and row stats ---
    for(std::size_t i = 0; i < h.m; ++i) {
        std::size_t rowStart = A.getRowOffset(i);
        std::size_t rowEnd   = A.getRowOffset(i + 1);

        std::uint32_t cnt = static_cast<std::uint32_t>(rowEnd - rowStart);
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

    std::size_t nnzBegin = A.getRowOffset(0);
    std::size_t nnzEnd   = A.getRowOffset(h.m);

    for(std::size_t k = nnzBegin; k < nnzEnd; ++k) {
        std::size_t j = A.getColIdx(k); // column index of this nnz
        std::uint32_t &cnt = h.hc[j];
        if(cnt == 0)
            h.nnzCols++;
        cnt++;
    }

    for(std::size_t j = 0; j < h.n; ++j) {
        std::uint32_t cnt = h.hc[j];
        if(cnt == 1)
            h.colsEq1++;
        if(cnt > h.m / 2)
            h.colsGtHalf++;
        if(cnt > h.maxHc)
            h.maxHc = cnt;
    }

    return h;
}
