
#include "rowbins.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

template<class T>
__global__ void rowBinningKernel(CCMatrix<T>* mat, uint32_t* bin_sizes, uint32_t* bins, const size_t* row_ptrs, size_t num_rows) {
    auto rid = blockIdx.x * blockDim.x + threadIdx.x;
//    uint32_t r_nnz = 0;
    
    if(rid < num_rows) {
        auto bin = 0;
        MatrixAccessor<T> A(mat);
        auto r_nnz = row_ptrs[rid + 1] - row_ptrs[rid];
        
        if (r_nnz > 0) {
            if(r_nnz <= 16)
                bin = 1;
            else if(r_nnz <= 32)
                bin = 2;
            else if(r_nnz <= 64)
                bin = 3;
            else if(r_nnz <= 128)
                bin = 4;
            else if(r_nnz <= 256)
                bin = 5;
            else if(r_nnz <= 512)
                bin = 6;
            else
                bin = 7;
        }
        auto idx = atomicAdd(&bin_sizes[bin], 1);
        //        printf("row %d nnz=%d bin=%d idx=%d gidx=%lu\n", rid, r_nnz, bin, idx, bin*num_rows+idx);
        bins[bin * num_rows + idx] = rid;
    }
}

template<class T>
void rowBins(CCMatrix<T>* mat, uint32_t* bin_sizes, uint32_t* bins, const size_t* row_ptrs, size_t num_rows) {
    auto blockSize = 256;
    auto gridSize = (num_rows + blockSize - 1) / blockSize;
    rowBinningKernel<T><<<gridSize, blockSize>>>(mat, bin_sizes, bins, row_ptrs, num_rows);
}

template
void rowBins<double>(CCMatrix<double>* mat, uint32_t* bin_sizes, uint32_t* bins, const size_t* row_ptrs, size_t num_rows);

template
void rowBins<float>(CCMatrix<float>* mat, uint32_t* bin_sizes, uint32_t* bins, const size_t* row_ptrs, size_t num_rows);

template
void rowBins<long>(CCMatrix<long>* mat, uint32_t* bin_sizes, uint32_t* bins, const size_t* row_ptrs, size_t num_rows);