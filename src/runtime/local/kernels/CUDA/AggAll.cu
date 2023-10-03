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

#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

#include "AggAll.h"
#include "HostUtils.h"
#include "bin_ops.cuh"
#include "DeviceUtils.cuh"

namespace CUDA {

    template<class VT, class ReductionOp, class AssigmentOp>
    __global__ void agg(VT* output, const VT* input, size_t N) {
        auto sdata = shared_memory_proxy<VT>();

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        unsigned int gridSize = blockDim.x * 2 * gridDim.x;

        VT v = ReductionOp::init();

        // we reduce multiple elements per thread.  The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        while (i < N) {
            v = ReductionOp::exec(v, input[i]);
            // ensure we don't read out of bounds
            if (i + blockDim.x < N)
                v = ReductionOp::exec(v, input[i + blockDim.x]);
            i += gridSize;
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

        // write result for this block to global mem
        if (tid == 0)
            output[blockIdx.x] = sdata[0];
    }

    template<typename VTRes, typename VTArg>
    VTRes AggAll<VTRes, DenseMatrix<VTArg>>::apply(AggOpCode opCode, const DenseMatrix<VTArg> * arg, DCTX(dctx)) {

        const size_t deviceID = 0; //ToDo: multi device support
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

        int blockSize;
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        size_t gridSize;

        auto N = arg->getNumItems();
        VTRes result;
        VTRes* tmp_ptr;

        // determine kernel launch parameters
        CHECK_CUDART(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize,
                agg<VTRes, SumOp<VTArg>, IdentityOp<VTRes>>, smem_calc<VTArg>(), 0));
        gridSize = (N + blockSize - 1) / blockSize;
        auto shmSize = sizeof(VTArg) * blockSize;

        // ToDo: get rid of this per-invocation malloc/free
        CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&tmp_ptr), blockSize * sizeof(VTRes)));

        if (opCode == AggOpCode::SUM) {
            agg<VTRes, SumOp<VTArg>,  IdentityOp<VTRes>><<<gridSize, blockSize, shmSize>>>(tmp_ptr,
                    arg->getValues(&alloc_desc), N);
        }
        else if (opCode == AggOpCode::MIN) {
            agg<VTRes, MinOp<VTArg>, IdentityOp<VTRes>><<<gridSize, blockSize, shmSize>>>(tmp_ptr,
                    arg->getValues(&alloc_desc), N);
        }
        else if (opCode == AggOpCode::MAX) {
            agg<VTRes, MaxOp<VTArg>, IdentityOp<VTRes>><<<gridSize, blockSize, shmSize>>>(tmp_ptr,
                    arg->getValues(&alloc_desc), N);
        }
        else {
            throw std::runtime_error(fmt::format("Unknown opCode {} for aggCol", static_cast<uint32_t>(opCode)));
        }
        CHECK_CUDART(cudaMemcpy(reinterpret_cast<void **>(&result), tmp_ptr, sizeof(VTRes), cudaMemcpyDeviceToHost));
        CHECK_CUDART(cudaFree(tmp_ptr));

        return result;
    }

    template struct AggAll<double, DenseMatrix<double>>;
    template struct AggAll<float, DenseMatrix<float>>;
    template struct AggAll<int64_t, DenseMatrix<int64_t>>;
}

