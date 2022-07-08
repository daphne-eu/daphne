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

#include "Syrk.h"

#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

namespace CUDA {
    template<typename T>
    __global__ void copy_u2l_dense(T *ret, int dim, int N) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto ix = tid / dim;
        auto iy = tid % dim;
        auto id_dest = iy * dim + ix;
        if(iy > ix && id_dest < N) {
            ret[id_dest] = ret[tid];
        }
    }

    template<typename T>
    void launch_cublas_syrk(const CUDAContext &ctx, size_t nr1, size_t nc1, const T *alpha, const T *beta,
            const T *d_arg, T *d_res);

    template<>
    [[maybe_unused]] void launch_cublas_syrk<double>(const CUDAContext &ctx, size_t nr1, size_t nc1, const double *alpha,
            const double *beta, const double *d_arg, double *d_res) {
        CHECK_CUBLAS(cublasDsyrk_v2(ctx.getCublasHandle(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nc1, nr1, alpha, d_arg,
                nc1, beta, d_res, nc1));
    }

    template<>
    [[maybe_unused]] void launch_cublas_syrk<float>(const CUDAContext &ctx, size_t nr1, size_t nc1,
            const float *alpha, const float *beta, const float *d_arg, float *d_res) {
        CHECK_CUBLAS(cublasSsyrk_v2(ctx.getCublasHandle(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nc1, nr1, alpha, d_arg,
                nc1, beta, d_res, nc1));
    }

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------
    template<typename VTres, typename VTarg>
    void Syrk<DenseMatrix<VTres>, DenseMatrix<VTarg>>::apply(DenseMatrix <VTres> *&res, const DenseMatrix <VTarg> *arg,
            DCTX(dctx)) {
        using VT = typename DenseMatrix<VTres>::VT;
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        const size_t nr1 = arg->getNumRows();
        const size_t nc1 = arg->getNumCols();

        const VT blend_alpha = 1.0f;
        const VT blend_beta = 0.0f;
        const VT *d_arg = arg->getValues(&alloc_desc);

        if(res == nullptr)
            res = DataObjectFactory::create < DenseMatrix < VTres >> (nc1, nc1, false, &alloc_desc);
        
        VT *d_res = res->getValues(&alloc_desc);
        launch_cublas_syrk<VT>(*ctx, nr1, nc1, &blend_alpha, &blend_beta, d_arg, d_res);

        // num threads
        int NT = 256;
        // num data items
        auto N = nc1 * nc1;
        // values per thread
        int vpt = 1;
        // num blocks
        std::uint32_t NB = std::ceil((N + NT * vpt - 1) / (NT * vpt));

        dim3 grid(NB, 1, 1);
        dim3 block(NT, 1, 1);

        copy_u2l_dense<<<grid, block>>>(d_res, nc1, N);
    }

    template struct Syrk<DenseMatrix<double>, DenseMatrix<double>>;
    template struct Syrk<DenseMatrix<float>, DenseMatrix<float>>;
}