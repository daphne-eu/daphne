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

#include <compiler/codegen/spoof-launcher/SpoofCellwise.h>
#include <compiler/codegen/spoof-launcher/SpoofRowwise.h>
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>
#include "runtime/local/datastructures/CSRMatrix.h"
#include "runtime/local/datastructures/DenseMatrix.h"

#include <iostream>

template<typename VT>
uint32_t writeInputToStagingBuffer2(std::byte* bbuf, const DenseMatrix<VT>* input, const uint32_t start_pos,
        AllocationDescriptorCUDA* alloc_desc) {
    uint32_t pos = start_pos;
    size_t tmp;
    auto ptr_size_t = reinterpret_cast<size_t*>(&(bbuf[pos]));
    *ptr_size_t = input->getNumItems();
    ptr_size_t++;
    auto ptr = reinterpret_cast<uint32_t*>(ptr_size_t);
    tmp = input->getNumRows();
    *ptr = *reinterpret_cast<uint32_t*>(&tmp);
    ptr++;
    tmp = input->getNumCols();
    *ptr = *reinterpret_cast<uint32_t*>(&tmp);
    ptr++;

    // row_ptr
    *ptr = 0;
    ptr++;
    // col_idxs
    *ptr = 0;
    ptr++;

    auto vtmp = input->getValues(alloc_desc);
    auto vtmp1 =  const_cast<double*>(vtmp);
    auto vtmp2 = reinterpret_cast<size_t>(vtmp1);
    std::cout << vtmp2 << std::endl;
    ptr_size_t = reinterpret_cast<size_t*>(ptr);
    *ptr_size_t = vtmp2;
    pos += (sizeof(size_t) + 4 * sizeof(uint32_t*) + sizeof(VT));
    return pos;
}

template<typename VT>
uint32_t writeInputToStagingBuffer(std::byte* bbuf, const DenseMatrix<VT>* input, const uint32_t start_pos,
                                   AllocationDescriptorCUDA* alloc_desc) {
    uint32_t pos = start_pos;
    size_t tmp;
    auto ptr64 = reinterpret_cast<uint64_t*>(&(bbuf[pos]));
    *ptr64 = input->getNumItems();
    pos += sizeof(uint64_t);
//    ptr_size_t++;
    auto ptr = reinterpret_cast<uint32_t*>(&(bbuf[pos]));
    tmp = input->getNumRows();
    *ptr = *reinterpret_cast<uint32_t*>(&tmp);
//    ptr++;
    pos += sizeof(uint32_t);
    ptr = reinterpret_cast<uint32_t*>(&(bbuf[pos]));
    tmp = input->getNumCols();
    *ptr = *reinterpret_cast<uint32_t*>(&tmp);
//    ptr++;
    pos += sizeof(uint32_t);

    // row_ptr
    ptr = reinterpret_cast<uint32_t*>(&(bbuf[pos]));
    ptr = 0;
//    ptr++;
    pos += sizeof(uint32_t*);

    // col_idxs
    ptr = reinterpret_cast<uint32_t*>(&(bbuf[pos]));
    ptr = 0;
//    ptr++;
    pos += sizeof(uint32_t*);

    auto vtmp = input->getValues(alloc_desc);
    auto vtmp1 =  const_cast<VT*>(vtmp);
    auto vtmp2 = reinterpret_cast<size_t>(vtmp1);
//    std::cout << vtmp2 << std::endl;
    ptr64 = reinterpret_cast<size_t*>(&(bbuf[pos]));
    *ptr64 = vtmp2;
    pos += sizeof(VT*);
//    pos += (sizeof(size_t) + 4 * sizeof(uint32_t*) + sizeof(VT));
    return pos;
}

template<typename VTres, typename VTarg>
VTres CodeGenCW(const DenseMatrix<VTarg>** args, DCTX(dctx)) {
    const size_t deviceID = 0; //ToDo: multi device support
    if(dctx->useCUDA())
        dctx->logger->info("daphne context CodeGenCW");

    auto ctx = CUDAContext::get(dctx, deviceID);
    AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

    auto a = args[0];
    auto b = args[1];

    auto cctx = reinterpret_cast<SpoofCUDAContext*>(dctx->getUserConfig().codegen_ctx_ptr);
    auto opID = reinterpret_cast<size_t>(args[2]);
    auto operator_name = cctx->getOperatorName(opID);
    std::cout << "executing op=" << operator_name << " id=" << opID << std::endl;

    int32_t num_inputs = 1;
    int32_t num_side_inputs = 1;
    int32_t num_scalars = 0;
    int32_t grix = 0;
    int32_t is_agg = 0;
    auto buf = reinterpret_cast<int32_t*>(cctx->staging_buffer);
    buf[0] = (num_inputs + num_side_inputs + 1) * JNI_MAT_ENTRY_SIZE + num_scalars * sizeof(VTarg) + TRANSFERRED_DATA_HEADER_SIZE;
    buf[1] = static_cast<int32_t>(opID);
    buf[2] = grix;
    buf[3] = num_inputs;
    buf[4] = num_side_inputs;
    buf[5] = is_agg;
    buf[6] = num_scalars;
    buf[7] = -1;

    ctx->logger->debug("sizeof(uint32_t)={}, sizeof(uint32_t*)={}, sizeof(size_t)={}, sizeof(size_t*)={}",
                       sizeof(uint32_t), sizeof(uint32_t*), sizeof(size_t), sizeof(size_t*));
    // transfers resource pointers to GPU
    auto pos = writeInputToStagingBuffer(cctx->staging_buffer, a, TRANSFERRED_DATA_HEADER_SIZE, &alloc_desc);
    pos = writeInputToStagingBuffer(cctx->staging_buffer, b, pos, &alloc_desc);

    ctx->logger->debug("pos before aggallout: {}", pos);
    pos += (2 * sizeof(uint32_t) + 2 * sizeof(uint32_t*) + sizeof(uint64_t));
    auto d_res = reinterpret_cast<VTres*>(ctx->getScratchBuffer());
    auto ptr64 = reinterpret_cast<size_t*>(&(cctx->staging_buffer[pos]));
    *ptr64 = reinterpret_cast<size_t>(d_res);
    pos += sizeof(size_t);

    ctx->logger->debug("codeGenCW: wrote {} bytes in staging buffer", pos);
    // launch gen op
    cctx->launch<VTarg, SpoofCellwise<VTarg>>();

    ctx->logger->debug("returned from gen_cw op");

    CHECK_CUDART(cudaStreamSynchronize(cctx->stream));

    ctx->logger->debug("stream synchronized");

//    std::vector<VTres> res(1048576);
//    CHECK_CUDART(cudaMemcpy(res.data(), d_res, sizeof(VTres) * 24, cudaMemcpyDeviceToHost));
//    ctx->logger->debug("cw full agg returning:");
//    for(auto r = 0; r < 24; ++r) {
//        ctx->logger->debug("res[{}]={}", r, res[r]);
//    }
//    return res[0];

    VTres res;
    CHECK_CUDART(cudaMemcpy(&res, d_res, sizeof(VTres), cudaMemcpyDeviceToHost));
    return res;
}

template<typename VTres, typename VTarg>
void CodeGenRW(DenseMatrix<VTres>*& res, const DenseMatrix<VTarg>** args, DCTX(dctx)) {
    const size_t deviceID = 0; //ToDo: multi device support
    if(dctx->useCUDA())
        dctx->logger->info("daphne context CodeGenRW");

    auto ctx = CUDAContext::get(dctx, deviceID);
    AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

    auto a = args[0];
    auto b = args[1];

    if(res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<VTres>>(a->getNumRows(), 1, false, &alloc_desc);

    auto cctx = reinterpret_cast<SpoofCUDAContext*>(dctx->getUserConfig().codegen_ctx_ptr);
    auto opID = reinterpret_cast<size_t>(args[2]);
    auto operator_name = cctx->getOperatorName(opID);
    std::cout << "executing op=" << operator_name << " id=" << opID << std::endl;

    int32_t num_inputs = 1;
    int32_t num_side_inputs = 1;
    int32_t num_scalars = 0;
    int32_t grix = 0;
    int32_t is_agg = 0;
    auto buf = reinterpret_cast<int32_t*>(cctx->staging_buffer);
    buf[0] = (num_inputs + num_side_inputs + 1) * JNI_MAT_ENTRY_SIZE + num_scalars * sizeof(VTarg) + TRANSFERRED_DATA_HEADER_SIZE;
    buf[1] = static_cast<int32_t>(opID);
    buf[2] = grix;
    buf[3] = num_inputs;
    buf[4] = num_side_inputs;
    buf[5] = is_agg;
    buf[6] = num_scalars;
    buf[7] = -1;

    ctx->logger->debug("sizeof(uint32_t)={}, sizeof(uint32_t*)={}, sizeof(size_t)={}, sizeof(size_t*)={}",
                sizeof(uint32_t), sizeof(uint32_t*), sizeof(size_t), sizeof(size_t*));
    // transfers resource pointers to GPU
    auto pos = writeInputToStagingBuffer(cctx->staging_buffer, a, TRANSFERRED_DATA_HEADER_SIZE, &alloc_desc);
    pos = writeInputToStagingBuffer(cctx->staging_buffer, b, pos, &alloc_desc);
    pos = writeInputToStagingBuffer(cctx->staging_buffer, res, pos, &alloc_desc);

    ctx->logger->debug("codeGenRW: wrote {} bytes in staging buffer", pos);
    // launch gen op
    cctx->launch<VTarg, SpoofRowwise<VTarg>>();

    ctx->logger->debug("returned from gen_rw op");

    CHECK_CUDART(cudaStreamSynchronize(cctx->stream));

    ctx->logger->debug("stream synchronized");
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


    void CUDA_codegenCW__double__DenseMatrix_double_variadic__size_t(double *res, const DenseMatrix<double> **arg,
            size_t num_operands, DCTX(ctx)) {
        auto nc = ctx->cuda_contexts.size();
        std::cout << "codegenCW: num operands: " << num_operands << ", num cuda contexts: " << nc << std::endl;
        *res = CodeGenCW<double>(arg, ctx);
    }

    void CUDA_codegenRW__DenseMatrix_double__DenseMatrix_double_variadic__size_t(DenseMatrix<double> **res,
            const DenseMatrix<double> **arg, size_t num_operands, DCTX(ctx)) {
        auto nc = ctx->cuda_contexts.size();
        std::cout << "codegenRW: num operands: " << num_operands << ", num cuda contexts: " << nc << std::endl;
        CodeGenRW(*res, arg, ctx);
    }

    void CUDA_codegenRW__DenseMatrix_int64_t__CSRMatrix_int64_t_variadic__size_t(DenseMatrix<int64_t> **res,
            const CSRMatrix<int64_t > **arg, DCTX(ctx)) {
        CodeGenRW(*res, arg, ctx);
    }
}