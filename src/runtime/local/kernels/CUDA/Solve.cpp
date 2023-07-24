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

// https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples

#include "Solve.h"
#include "runtime/local/kernels/CUDA/Transpose.h"

namespace CUDA {
// -----------------------------------------------------------------------------------------------------------------
    template<typename T>
    cusolverStatus_t cusolverDnXgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, T *A, int lda, int *Lwork);

    template<>
    [[maybe_unused]] cusolverStatus_t
    cusolverDnXgetrf_bufferSize<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda,
                                        int *Lwork) {
        return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }

    template<>
    [[maybe_unused]] cusolverStatus_t
    cusolverDnXgetrf_bufferSize<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda,
                                       int *Lwork) {
        return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }

// -----------------------------------------------------------------------------------------------------------------
    template<typename T>
    cusolverStatus_t
    cusolverDnXgetrf(cusolverDnHandle_t handle, int m, int n, T *A, int lda, T *Workspace, int *devIpiv, int *devInfo);

    template<>
    [[maybe_unused]] cusolverStatus_t
    cusolverDnXgetrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace,
                             int *devIpiv, int *devInfo) {
        return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    }

    template<>
    [[maybe_unused]] cusolverStatus_t
    cusolverDnXgetrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace,
                            int *devIpiv, int *devInfo) {
        return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    }

// -----------------------------------------------------------------------------------------------------------------
    template<typename T>
    cusolverStatus_t
    cusolverDnXgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const T *A, int lda,
                     const int *devIpiv, T *B, int ldb, int *devInfo);

    template<>
    [[maybe_unused]] cusolverStatus_t
    cusolverDnXgetrs<double>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
                             const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo) {
        return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
    }

    template<>
    [[maybe_unused]] cusolverStatus_t
    cusolverDnXgetrs<float>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
                            const float *A, int lda, const int *devIpiv, float *B, int ldb, int *devInfo) {
        return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
    }

    template<class VT>
    void Solve<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>>::apply
            (DenseMatrix<VT> *&res, const DenseMatrix<VT> *lhs, const DenseMatrix<VT> *rhs, DCTX(dctx)) {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        
        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nc2 = rhs->getNumCols();
        assert((nr1 == rhs->getNumRows()) && "#rows of lhs and #rows of rhs must be the same");
        assert((nr1 == nc1) && "#rows and #cols of lhs must be the same");
        assert((lhs->getRowSkip() == nc1) && "#cols of lhs must match row skip");
        assert((nc2 == 1) && "#cols of rhs must be 1");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false, &alloc_desc);

        int *d_Ipiv{}; /* pivoting sequence */
        int *d_info{}; /* error info */
        int lwork = 0;     /* size of workspace */
        VT *d_work{}; /* device workspace for getrf */
        VT *d_A{};
        CHECK_CUBLAS(cublasSetStream(ctx->getCublasHandle(), ctx->getCuSolverStream()));
        CHECK_CUDART(cudaMallocAsync(reinterpret_cast<void **>(&d_A), lhs->getBufferSize(), ctx->getCuSolverStream()));
        const VT blend_alpha = 1.0f;
        const VT blend_beta = 0.0f;

        launch_cublas_geam<VT>(*ctx, nr1, nc1, &blend_alpha, &blend_beta, lhs->getValues(&alloc_desc), d_A);
        CHECK_CUBLAS(cublasSetStream(ctx->getCublasHandle(), nullptr));
        auto &m = nc1;
        CHECK_CUDART(cudaMemcpyAsync(res->getValues(&alloc_desc), rhs->getValues(&alloc_desc), rhs->getBufferSize(),
                cudaMemcpyDeviceToDevice, ctx->getCuSolverStream()));
        auto d_B = res->getValues(&alloc_desc);
        auto lda = m;
        auto ldb = m;
        CHECK_CUDART(cudaMallocAsync(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * nr1, ctx->getCuSolverStream()));
        CHECK_CUDART(cudaMallocAsync(reinterpret_cast<void **>(&d_info), sizeof(int), ctx->getCuSolverStream()));

        //ToDo: templatize
        CHECK_CUSOLVER(cusolverDnXgetrf_bufferSize(ctx->getCUSOLVERHandle(), m, m, d_A, lda, &lwork));

        ctx->logger->debug("allocating {} bytes for cuSolver workspace", sizeof(VT) * lwork);

        CHECK_CUDART(cudaMallocAsync((void **) &d_work, sizeof(VT) * lwork, ctx->getCuSolverStream()));

        CHECK_CUSOLVER(cusolverDnXgetrf(ctx->getCUSOLVERHandle(), m, m, d_A, lda, d_work, d_Ipiv, d_info));

        int info = 0;     /* host copy of error info from cusolverDnXgetrf */
        CHECK_CUDART(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, ctx->getCuSolverStream()));
        CHECK_CUDART(cudaStreamSynchronize(ctx->getCuSolverStream()));
        if(info == 0) {
            CHECK_CUSOLVER(cusolverDnXgetrs(ctx->getCUSOLVERHandle(), CUBLAS_OP_N, m, nc2, d_A, lda, d_Ipiv, d_B, ldb,
                                            d_info));
            info = 0; // reset status and request outcome of cusolverDnXgetrs
            CHECK_CUDART(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, ctx->getCuSolverStream()));
            CHECK_CUDART(cudaStreamSynchronize(ctx->getCuSolverStream()));
            if(info > 0)
                throw std::runtime_error("cuSolve: A factor Ui is exactly singular, so the solution could not be computed");
            else if(info < 0)
                throw std::runtime_error(std::string(std::string("cuSolve: The ") + std::to_string(info) +
                        std::string("-th value had an illegal value.")).c_str());
        }
        else if(info > 0)
            throw std::runtime_error("cuSolve: A factor Ui is exactly singular, so the solution could not be computed");
        else if(info < 0)
            throw std::runtime_error(std::string(std::string("cuSolve: The ") + std::to_string(info) +
                    std::string("-th value had an illegal value.")).c_str());
        cudaFreeAsync(d_A, ctx->getCuSolverStream());
    }

    template struct Solve<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
    template struct Solve<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
}