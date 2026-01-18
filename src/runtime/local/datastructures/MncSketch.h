#pragma once

#include <iostream>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <numeric>   
#include <algorithm> 

#include <runtime/local/datastructures/CSRMatrix.h>

struct MncSketch{
    // dimensions
    std::size_t m = 0; // rows
    std::size_t n = 0; // cols

    // core counts
    std::vector<std::uint32_t> hr;   // nnz per row (size m)
    std::vector<std::uint32_t> hc;   // nnz per col (size n)

    // Extended counts (optional, only constructed if maxHr or maxHc > 1)
    // her[i]: nnz in row i that lie in columns with hc == 1
    // hec[j]: nnz in column j that lie in rows with hr == 1
    std::vector<std::uint32_t> her;
    std::vector<std::uint32_t> hec;

    // Summary statistics
    std::uint32_t maxHr = 0;
    std::uint32_t maxHc = 0;
    std::uint32_t nnzRows = 0;       // # n rows with hr > 0
    std::uint32_t nnzCols = 0;       // # n cols with hc > 0
    std::uint32_t rowsEq1 = 0;       // # n rows with hr == 1, 
    std::uint32_t colsEq1 = 0;       // # n cols with hc == 1
    std::uint32_t rowsGtHalf = 0;    // # n rows with hr > n/2
    std::uint32_t colsGtHalf = 0;    // # n cols with hc > m/2
    bool isDiagonal;         // optional flag if A is (full) diagonal
};

// Build MNC sketch from a CSRMatrix
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

    // --- 3) isDiagonal ---
    // We call a matrix "diagonal" if it is square and every non-zero lies on i == j.
    if(h.m == h.n && nnzEnd > nnzBegin) {
        bool diag = true;
        for(std::size_t i = 0; i < h.m && diag; ++i) {
            std::size_t s = rowOffsets[i];
            std::size_t e = rowOffsets[i+1];
            for(std::size_t k = s; k < e; ++k) {
                std::size_t j = colIdxs[k];
                if(j != i) {
                    diag = false;
                    break;
                }
            }
        }
        h.isDiagonal = diag;
    } else {
        h.isDiagonal = false;
    }

    // --- 4) extended counts her, hec --- (only if there is something to extend)
    if(h.maxHr > 1 || h.maxHc > 1) {
        h.her.assign(h.m, 0);
        h.hec.assign(h.n, 0);

        // For each nnz at (i,j):
        //  - if hc[j] == 1, it contributes to her[i]
        //  - if hr[i] == 1, it contributes to hec[j]
        for(std::size_t i = 0; i < h.m; ++i) {
            std::size_t s = rowOffsets[i];
            std::size_t e = rowOffsets[i+1];
            for(std::size_t k = s; k < e; ++k) {
                std::size_t j = colIdxs[k];

                if(h.hc[j] == 1)
                    h.her[i]++;

                if(h.hr[i] == 1)
                    h.hec[j]++;
            }
        }
    }

    return h;
}

// TODO: implement the EDM algorithm properly
inline double Edm(const std::vector<std::uint32_t>&,
                  const std::vector<std::uint32_t>&,
                  std::size_t) {
    return 0.5; // placeholder 
}



double estimateSparsity_product(const MncSketch &hA, const MncSketch &hB) {
    const std::size_t m = hA.m;
    const std::size_t l = hB.n;

    double nnz = 0.0;

    // Case 1: Exact count
    if(hA.maxHr <= 1 || hB.maxHc <= 1) {
        for(std::size_t j = 0; j < hA.n; ++j)
            nnz += static_cast<double>(hA.hc[j]) * static_cast<double>(hB.hr[j]);
    }

    // Case 2: Extended count
    else if(!hA.her.empty() || !hB.her.empty()) {

        // Exact part
        for(std::size_t j = 0; j < hA.n; ++j)
            nnz += static_cast<double>(hA.hec[j]) * static_cast<double>(hB.hr[j]);

        for(std::size_t i = 0; i < hB.m; ++i)
            nnz += static_cast<double>(hB.her[i]) * static_cast<double>(hA.hc[i] - hA.hec[i]);

        // Remaining uncertain cells
        std::size_t p = (hA.nnzRows - hA.rowsEq1) * (hB.nnzCols - hB.colsEq1);

        if(p > 0) {
            double dens = Edm(hA.hc, hB.hr, p);
            nnz += dens * static_cast<double>(p);
        }
    }

    // Case 3: Generic fallback
    else {
        std::size_t p = hA.nnzRows * hB.nnzCols;
        if(p > 0) {
            double dens = Edm(hA.hc, hB.hr, p);
            nnz = dens * static_cast<double>(p);
        }
    }

    // Lower bound
    std::size_t lower = hA.rowsGtHalf * hB.colsGtHalf;

    if(nnz < static_cast<double>(lower))
        nnz = static_cast<double>(lower); 

    return nnz / static_cast<double>(m * l); 
}