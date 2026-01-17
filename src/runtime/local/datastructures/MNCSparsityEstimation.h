#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <numeric>   
#include <algorithm> 

#include "MncSketch.h"


inline double Edm(const std::vector<std::uint32_t>&,
                  const std::vector<std::uint32_t>&,
                  std::size_t) {
    return 0.5; // placeholder 
}



double estimateSparsity(const MncSketch &hA, const MncSketch &hB) {
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
