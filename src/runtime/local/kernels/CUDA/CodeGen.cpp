/*
 * Copyright 2023 The DAPHNE Consortium
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


#include <compiler/codegen/spoof-launcher/SpoofCUDAContext.h>
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>
#include "runtime/local/datastructures/CSRMatrix.h"
#include "runtime/local/datastructures/DenseMatrix.h"

#include <iostream>

template<typename VTres, typename VTarg>
void CodeGenRW(DenseMatrix<VTres>*& res, const DenseMatrix<VTarg>** args, DCTX(dctx)) {
    const size_t deviceID = 0; //ToDo: multi device support
    if(dctx->useCUDA())
        dctx->logger->info("daphne context CodeGenRW");

    auto ctx = CUDAContext::get(dctx, deviceID);
    AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

    // get codegen context

    auto a = args[0];
    if(res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<VTres>>(a->getNumRows(), 1, false, &alloc_desc);

    // launch gen op
    ctx->logger->debug("launch CodeGenRW kernel");
}


template<typename VTres, typename VTarg>
void CodeGenRW(DenseMatrix<VTres>*& res, const CSRMatrix<VTarg>** args, DCTX(dctx)) {
    const size_t deviceID = 0; //ToDo: multi device support
    if(dctx)
        dctx->logger->info("daphne context CodeGenRW");

    auto ctx = CUDAContext::get(dctx, deviceID);
    AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

    // get codegen context

    auto a = args[0];
    auto b = args[1];

    SpoofCUDAContext* cctx = reinterpret_cast<SpoofCUDAContext*>(dctx->getUserConfig().codegen_ctx_ptr);
//    cctx->template launch<SpoofRowwiseOp>()

    if(res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<VTres>>(a->getNumRows(), 1, false, &alloc_desc);

    // launch gen op
    ctx->logger->debug("launch CodeGenRW kernel");
}

extern "C" {
        void CUDA_codegenRW__DenseMatrix_double__DenseMatrix_double_variadic__size_t(DenseMatrix<double> **res,
                const DenseMatrix<double> **arg, size_t num_operands, DCTX(ctx)) {
            auto nc = ctx->cuda_contexts.size();
            std::cout << "num operands: " << num_operands << ", num cuda contexts: " << nc << std::endl;
            CodeGenRW(*res, arg, ctx);
        }

        void CUDA_codegenRW__DenseMatrix_int64_t__CSRMatrix_int64_t_variadic__size_t(DenseMatrix<int64_t> **res, const CSRMatrix<int64_t > **arg, DCTX(ctx)) {
        CodeGenRW(*res, arg, ctx);
    }
}