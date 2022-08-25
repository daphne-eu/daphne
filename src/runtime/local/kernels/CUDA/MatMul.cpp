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

#include "MatMul.h"
#include "Gemv.h"
#include "runtime/local/datastructures/AllocationDescriptorCUDA.h"

namespace CUDA {

    template<typename T>
    void launch_cublas_gemm(const CUDAContext &ctx, size_t nr1, size_t nc1, size_t nc2, const T *alpha, const T *beta,
                            const T *d_lhs, const T *d_rhs, T *d_res);

    template<>
    [[maybe_unused]] void launch_cublas_gemm<float>(const CUDAContext &ctx, size_t nr1, size_t nc1, size_t nc2,
                                                    const float *alpha, const float *beta, const float *d_lhs,
                                                    const float *d_rhs, float *d_res) {
        CHECK_CUBLAS(
                cublasSgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
                            nc1, beta, d_res, nc2));
    }

    template<>
    [[maybe_unused]] void launch_cublas_gemm<double>(const CUDAContext &ctx, size_t nr1, size_t nc1, size_t nc2,
                                                     const double *alpha, const double *beta, const double *d_lhs,
                                                     const double *d_rhs, double *d_res) {
        CHECK_CUBLAS(
                cublasDgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
                            nc1, beta, d_res, nc2));
    }

    template<typename T>
    void MatMul<DenseMatrix<T>, DenseMatrix<T>, DenseMatrix<T>>::apply(DenseMatrix<T> *&res, const DenseMatrix<T> *lhs,
                                                                       const DenseMatrix<T> *rhs, DCTX(dctx)) {
        using VT = typename DenseMatrix<T>::VT;
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nc2 = rhs->getNumCols();
        assert((nc1 == rhs->getNumRows()) && "#cols of lhs and #rows of rhs must be the same");
        const VT blend_alpha = 1.0f;
        const VT blend_beta = 0.0f;
        const VT *d_lhs = lhs->getValues(&alloc_desc);
        const VT *d_rhs = rhs->getValues(&alloc_desc);
    
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<T>>(nr1, nc2, false, &alloc_desc);
        VT *d_res = res->getValues(&alloc_desc);

        if(nc2 == 1) {
            launch_cublas_gemv<VT>(*ctx, nc1, nr1, &blend_alpha, &blend_beta, d_lhs, d_rhs,
                                   d_res, CUBLAS_OP_T);
        }
        else {
//             reverse order to accommodate cublas' col major format (-> res = rhs * lhs)
            launch_cublas_gemm<VT>(*ctx, nr1, nc1, nc2, &blend_alpha, &blend_beta, d_lhs, d_rhs, d_res);
        }
    }


    //ToDo: sparse mat mult (sample code below compiles but is not usable)
// from cusparse sample code:
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spgemm/spgemm_example.c
    template<typename T>
    void MatMul<CSRMatrix<T>, CSRMatrix<T>, CSRMatrix<T>>::apply(CSRMatrix<T> *&res, const CSRMatrix<T> *lhs,
                                                                 const CSRMatrix<T> *rhs, DCTX(dctx)) {
        using VT = typename DenseMatrix<T>::VT;
        auto ctx = CUDAContext::get(dctx, 0);
        cusparseHandle_t handle = ctx->getCusparseHandle();

        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();
        const size_t A_nnz = lhs->getNumNonZeros();
        const size_t B_nnz = rhs->getNumNonZeros();
        assert((nc1 == nr2) && "#cols of lhs and #rows of rhs must be the same");
        const VT blend_alpha = 1.0f;
        const VT blend_beta = 0.0f;
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cudaDataType computeType = ctx->template getCUSparseDataType<VT>();

        //--------------------------------------------------------------------------
        // Device memory management: Allocate and copy A, B
        int *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns, *dC_csrOffsets, *dC_columns;
        VT *dA_values, *dB_values, *dC_values;

        // allocate A
        CHECK_CUDART(cudaMalloc((void **) &dA_csrOffsets, (nr1 + 1) * sizeof(int)));
        CHECK_CUDART(cudaMalloc((void **) &dA_columns, A_nnz * sizeof(int)));
        CHECK_CUDART(cudaMalloc((void **) &dA_values, A_nnz * sizeof(VT)));

        // allocate B
        CHECK_CUDART(cudaMalloc((void **) &dB_csrOffsets, (nr2 + 1) * sizeof(int)));
        CHECK_CUDART(cudaMalloc((void **) &dB_columns, B_nnz * sizeof(int)));
        CHECK_CUDART(cudaMalloc((void **) &dB_values, B_nnz * sizeof(VT)));

        // allocate C offsets
        CHECK_CUDART(cudaMalloc((void **) &dC_csrOffsets, (nr1 + 1) * sizeof(int)));

        // copy A
        CHECK_CUDART(cudaMemcpy(dA_csrOffsets, lhs->getRowOffsets(), (nr1 + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDART(cudaMemcpy(dA_columns, lhs->getColIdxs(), A_nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDART(cudaMemcpy(dA_values, lhs->getValues(), A_nnz * sizeof(VT), cudaMemcpyHostToDevice));

        // copy B
        CHECK_CUDART(cudaMemcpy(dB_csrOffsets, rhs->getRowOffsets(), (nr2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDART(cudaMemcpy(dB_columns, rhs->getColIdxs(), B_nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDART(cudaMemcpy(dB_values, rhs->getValues(), B_nnz * sizeof(VT), cudaMemcpyHostToDevice));

        cusparseSpMatDescr_t matA, matB, matC;
        void *dBuffer1 = nullptr, *dBuffer2 = nullptr;
        size_t bufferSize1 = 0, bufferSize2 = 0;

        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE(
                cusparseCreateCsr(&matA, nr1, nc1, A_nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        CHECK_CUSPARSE(
                cusparseCreateCsr(&matB, nr2, nc2, B_nnz, dB_csrOffsets, dB_columns, dB_values, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        CHECK_CUSPARSE(cusparseCreateCsr(&matC, nr1, nc2, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        // SpGEMM Computation
        cusparseSpGEMMDescr_t spgemmDesc;
        CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

        // ask bufferSize1 bytes for external memory
        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &blend_alpha, matA, matB, &blend_beta, matC,
                                                     computeType,
                                                     CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr));

        CHECK_CUDART(cudaMalloc((void **) &dBuffer1, bufferSize1));

        // inspect the matrices A and B to understand the memory requirement for
        // the next step

        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &blend_alpha, matA, matB, &blend_beta, matC,
                                                     computeType,
                                                     CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));

        // ask bufferSize2 bytes for external memory
        CHECK_CUSPARSE(
                cusparseSpGEMM_compute(handle, opA, opB, &blend_alpha, matA, matB, &blend_beta, matC, computeType,
                                       CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr));

        CHECK_CUDART(cudaMalloc((void **) &dBuffer2, bufferSize2));

        // compute the intermediate product of A * B
        CHECK_CUSPARSE(
                cusparseSpGEMM_compute(handle, opA, opB, &blend_alpha, matA, matB, &blend_beta, matC, computeType,
                                       CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));

        // get matrix C non-zero entries C_nnz1
        int64_t C_num_rows1, C_num_cols1, C_nnz1;
        CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1));

        // allocate matrix C
        CHECK_CUDART(cudaMalloc((void **) &dC_columns, C_nnz1 * sizeof(int)));
        CHECK_CUDART(cudaMalloc((void **) &dC_values, C_nnz1 * sizeof(VT)));

        // update matC with the new pointers
        CHECK_CUSPARSE(cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values));

        // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

        // copy the final products to the matrix C
        CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &blend_alpha, matA, matB, &blend_beta, matC, computeType,
                                           CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        CHECK_CUSPARSE(cusparseDestroySpMat(matB));
        CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    }

    // explicit instantiations to satisfy linker
    template struct MatMul<CSRMatrix<float>, CSRMatrix<float>, CSRMatrix<float>>;
    template struct MatMul<CSRMatrix<double>, CSRMatrix<double>, CSRMatrix<double>>;
    template struct MatMul<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
    template struct MatMul<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
}