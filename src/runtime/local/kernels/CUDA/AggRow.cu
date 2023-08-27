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

#include "AggRow.h"
#include "HostUtils.h"

#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

#include "bin_ops.cuh"

#include "DeviceUtils.cuh"

namespace CUDA {

    template<class VT, class ReductionOp, class AssigmentOp>
    __global__ void agg_row(VT* output, const VT* input, size_t rows, size_t cols) {
        auto sdata = shared_memory_proxy<VT>();

        // one block per row
        if (blockIdx.x >= rows) {
            return;
        }

        unsigned int block = blockIdx.x;
        unsigned int tid = threadIdx.x;
        unsigned int i = tid;
        unsigned int block_offset = block * cols;

        VT v = ReductionOp::init();
        while (i < cols) {
            v = ReductionOp::exec(v, input[block_offset + i]);
            i += blockDim.x;
        }

        // each thread puts its local sum into shared memory
        sdata[tid] = v;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 1024) {
            if (tid < 512) {
                sdata[tid] = v = ReductionOp::exec(v, sdata[tid + 512]);
            }
            __syncthreads();
        }
        if (blockDim.x >= 512) {
            if (tid < 256) {
                sdata[tid] = v = ReductionOp::exec(v, sdata[tid + 256]);
            }
            __syncthreads();
        }
        if (blockDim.x >= 256) {
            if (tid < 128) {
                sdata[tid] = v = ReductionOp::exec(v, sdata[tid + 128]);
            }
            __syncthreads();
        }
        if (blockDim.x >= 128) {
            if (tid < 64) {
                sdata[tid] = v = ReductionOp::exec(v, sdata[tid + 64]);
            }
            __syncthreads();
        }

        if (tid < 32) {
            // now that we are using warp-synchronous programming (below)
            // we need to declare our shared memory volatile so that the compiler
            // doesn't reorder stores to it and induce incorrect behavior.
            volatile VT *smem = sdata;
            if (blockDim.x >= 64) {
                smem[tid] = v = ReductionOp::exec(v, smem[tid + 32]);
            }
            if (blockDim.x >= 32) {
                smem[tid] = v = ReductionOp::exec(v, smem[tid + 16]);
            }
            if (blockDim.x >= 16) {
                smem[tid] = v = ReductionOp::exec(v, smem[tid + 8]);
            }
            if (blockDim.x >= 8) {
                smem[tid] = v = ReductionOp::exec(v, smem[tid + 4]);
            }
            if (blockDim.x >= 4) {
                smem[tid] = v = ReductionOp::exec(v, smem[tid + 2]);
            }
            if (blockDim.x >= 2) {
                smem[tid] = v = ReductionOp::exec(v, smem[tid + 1]);
            }
        }

        // write result for this block to global mem, modify it with assignment op
        if (tid == 0)
            output[block] = AssigmentOp::exec(sdata[0], 0);
    }

    template<typename VT>
    void AggRow<DenseMatrix<VT>, DenseMatrix<VT>>::apply(AggOpCode opCode, DenseMatrix<VT> *&res,
            const DenseMatrix<VT> *arg, DCTX(dctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        const size_t deviceID = 0; //ToDo: multi device support
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        auto ctx = CUDAContext::get(dctx, deviceID);

        int threads;
        int blocks;

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, 1, false,  &alloc_desc);

        threads = (numCols < ctx->getMaxNumThreads() * 2) ? nextPow2((numCols + 1) / 2) : ctx->getMaxNumThreads();
        auto shmSize = sizeof(VT) * threads;
        blocks = numRows;

        if (opCode == AggOpCode::SUM) {
            agg_row<VT, SumOp<VT>, IdentityOp<VT>><<<blocks, threads, shmSize>>>(res->getValues(&alloc_desc),
                    arg->getValues(&alloc_desc), numRows, numCols);
        }
        else if (opCode == AggOpCode::MAX) {
            agg_row<VT, MaxOp<VT>, IdentityOp<VT>><<<blocks, threads, shmSize>>>(res->getValues(&alloc_desc),
                    arg->getValues(&alloc_desc), numRows, numCols);
        }
        else if (opCode == AggOpCode::MIN) {
            agg_row<VT, MinOp<VT>, IdentityOp<VT>><<<blocks, threads, shmSize>>>(res->getValues(&alloc_desc),
                    arg->getValues(&alloc_desc), numRows, numCols);
        }
        else {
            throw std::runtime_error(fmt::format("Unknown opCode {} for aggRow", static_cast<uint32_t>(opCode)));
        }
    }
    template struct AggRow<DenseMatrix<double>, DenseMatrix<double>>;
    template struct AggRow<DenseMatrix<float>, DenseMatrix<float>>;
    template struct AggRow<DenseMatrix<int64_t>, DenseMatrix<int64_t>>;
}

