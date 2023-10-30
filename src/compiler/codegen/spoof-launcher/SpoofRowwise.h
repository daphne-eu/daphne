/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include "SpoofCUDAContext.h"
#include <algorithm>
#include "rowbins.h"
#include <sstream>

using clk = std::chrono::high_resolution_clock;
using sec = std::chrono::duration<double, std::ratio<1>>;

template <typename T>
struct SpoofRowwise {

    static const uint32_t num_bins = 8;
    static const uint32_t NT = 256;

    static void exec([[maybe_unused]] SpoofCUDAContext* ctx, SpoofOperator* _op, DataBufferWrapper* dbw)  {
		T value_type;
		bool sparse_input = dbw->h_in<T>(0)->row_ptr != nullptr;
		auto* op = dynamic_cast<SpoofRowwiseOp*>(_op);
		dim3 grid(dbw->h_in<T>(0)->rows, 1, 1);
		dim3 block(NT, 1, 1);
		unsigned int shared_mem_size = NT * sizeof(T);

		size_t out_num_elements = dbw->h_out<T>()->rows * dbw->h_out<T>()->cols;
		if(dbw->h_out<T>()->row_ptr)
			if(op->isSparseSafe() && dbw->h_out<T>()->nnz > 0)
				out_num_elements = dbw->h_out<T>()->nnz;
		//ToDo: only memset output when there is an output operation that *adds* to the buffer
		CHECK_CUDART(cudaMemsetAsync(dbw->h_out<T>()->data, 0, out_num_elements * sizeof(T), ctx->stream));

		//ToDo: handle this in JVM
		uint32_t tmp_len = 0;
		uint32_t temp_buf_size = 0;
		T* d_temp = nullptr;
		if(op->num_temp_vectors > 0) {
			tmp_len = std::max(dbw->h_in<T>(0)->cols, op->const_dim2 < 0 ? 0u : static_cast<uint32_t>(op->const_dim2));
			temp_buf_size = op->num_temp_vectors * tmp_len * dbw->h_in<T>(0)->rows * sizeof(T);
#ifndef NDEBUG
			std::cout << "num_temp_vect: " << op->num_temp_vectors << " temp_buf_size: " << temp_buf_size << " tmp_len: " << tmp_len << std::endl;
#endif
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_temp), temp_buf_size));
			CHECK_CUDART(cudaMemsetAsync(d_temp, 0, temp_buf_size, ctx->stream));
		}

		std::string op_name;
        op_name = op->name + (sparse_input ? "_SPARSE" : "_DENSE");
#ifndef NDEBUG
        // ToDo: connect output to SystemDS logging facilities
        std::cout << "launching spoof rowwise kernel " << op_name << " with " << NT * dbw->h_in<T>(0)->rows
                  << " threads in "
                  << dbw->h_in<T>(0)->rows << " blocks and " << shared_mem_size << " bytes of shared memory for "
                  << dbw->h_in<T>(0)->rows << " cols processed by " << NT << " threads per row, adding "
                  << temp_buf_size / 1024 << " kb of temp buffer in global memory." << std::endl;
#endif

        auto binning_env = std::getenv("SYSDS_ROW_BINNING");
#ifndef NDEBUG
        if(binning_env)
            std::cout << "SYSDS_ROW_BINNING env var set to: " << binning_env << std::endl;
        else
            std::cout << "SYSDS_ROW_BINNING env var not set" << std::endl;
#endif

        if(sparse_input && binning_env != nullptr) {
            uint32_t* d_bins = nullptr;
            uint32_t* d_bin_sizes = nullptr;
            size_t bins_bufSize = dbw->h_in<T>(0)->rows * num_bins * sizeof(uint32_t);
            size_t bin_size_bufSize = num_bins * sizeof(uint32_t);
            CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_bins), bins_bufSize));
            CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_bin_sizes), bin_size_bufSize));
//            CHECK_CUDART(cudaMemsetAsync(d_bin_sizes, 0, bin_size_bufSize, ctx->stream));
            CHECK_CUDART(cudaMemset(d_bin_sizes, 0, bin_size_bufSize));
#ifndef NDEBUG
//            CHECK_CUDART(cudaDeviceSynchronize());
//            auto rpsize = (dbw->h_in<T>(0)->rows+1) * sizeof(uint32_t);
//            std::vector<uint32_t> h_row_ptrs(dbw->h_in<T>(0)->rows+1);
//            CHECK_CUDART(cudaMemcpy(h_row_ptrs.data(), dbw->h_in<T>(0)->row_ptr, rpsize, cudaMemcpyDeviceToHost));
//            std::cout << "ptrs:" << std::endl;
//            auto num_ptrs = 0;
//            for(const auto& p : h_row_ptrs) {
//                std::cout << "p[" << num_ptrs++ << "]=" << p << std::endl;
//            }

            std::cout << "nnz=" << dbw->h_in<T>(0)->nnz << std::endl;

//            std::vector<T> h_vals(dbw->h_in<T>(0)->nnz);
//            CHECK_CUDART(cudaMemcpy(h_vals.data(), dbw->h_in<T>(0)->data, dbw->h_in<T>(0)->nnz * sizeof(T), cudaMemcpyDeviceToHost));
//            auto num_vals = 0;
//            for(const auto& v : h_vals) {
//                std::cout << "v[" << num_vals++ << "]=" << v << std::endl;
//            }

#endif

            // launch binning
            rowBins(dbw->d_in<T>(0), d_bin_sizes, d_bins, dbw->h_in<T>(0)->row_ptr, dbw->h_in<T>(0)->rows);
            // copy bin_sizes back
            
            std::vector<uint32_t> h_bin_sizes(num_bins);
//            CHECK_CUDART(cudaMemcpyAsync(h_bin_sizes.data(), d_bin_sizes, bin_size_bufSize, cudaMemcpyDeviceToHost, ctx->stream));
            CHECK_CUDART(cudaMemcpy(h_bin_sizes.data(), d_bin_sizes, bin_size_bufSize, cudaMemcpyDeviceToHost));
#ifndef NDEBUG
            std::vector<uint32_t> h_bins(dbw->h_in<T>(0)->rows * num_bins);
//            CHECK_CUDART(cudaMemcpyAsync(h_bins.data(), d_bins, bins_bufSize, cudaMemcpyDeviceToHost, ctx->stream));
            CHECK_CUDART(cudaMemcpy(h_bins.data(), d_bins, bins_bufSize, cudaMemcpyDeviceToHost));
            std::cout << "bins:" << std::endl;
#endif
//            cudaStreamSynchronize(ctx->stream);

            // special bins:
            // bin0: empty rows
            // bin1: very sparse rows (<16 non zeros) -> 1 thread per row
            for (int i = 0; i < num_bins; i++) {
                auto bin_size = h_bin_sizes[i];

                // skip empty bins
                if(bin_size == 0) {
#ifndef NDEBUG
                    std::cout << "skipping empty bin " << i << std::endl;
#endif
                    continue;
                }
#ifndef NDEBUG
                std::cout << "bin[" << i << "]=" << bin_size << std::endl;

//                for (auto j = 0; j < bin_size; j++) {
//                    std::cout << h_bins[i * dbw->h_in<T>(0)->rows + j] << " ";
//                }
//                std::cout << std::endl;
#endif
                auto thread_bins = (bin_size + NT - 1) / NT;
//                auto thread_bins = bin_size;
                //launch binned kernels
                dim3 bin_grid(i < 2 ? thread_bins : bin_size, 1, 1);
                dim3 bin_block((i < 2 ? NT : 8 << i), 1, 1);
                unsigned int bin_shared_mem_size = (i > 1 ? bin_block.x * sizeof(T) : 0);

                std::string sparse_op_name = std::string(op_name + (i < 2 ? "_THREAD_BINS" : "_BLOCK_BINS"));
//                std::string sparse_op_name = std::string(op_name + "_BLOCK_BINS");
#ifndef NDEBUG
                    std::cout << "num_threads=" << bin_block.x << " num_blocks=" << bin_grid.x << " shared_mem=" << bin_shared_mem_size  << " bytes" << std::endl;
                    std::cout << sparse_op_name << std::endl;

                CHECK_CUDART(cudaDeviceSynchronize());
                auto start = clk::now();
#endif
                CHECK_CUDA(op->program->kernel(sparse_op_name)
                                   .instantiate(type_of(value_type),
                                                std::max(static_cast<uint32_t>(1), dbw->num_sides()),
                                                op->num_temp_vectors, tmp_len)
                                   .configure(bin_grid, bin_block, bin_shared_mem_size, ctx->stream)
                                   .launch(dbw->d_in<T>(0), dbw->d_sides<T>(), dbw->d_out<T>(), dbw->d_scalars<T>(),
                                           d_temp, dbw->grix(), d_bins, i, bin_size));
#ifndef NDEBUG
                cudaStreamSynchronize(ctx->stream);
                auto end = clk::now();
                auto duration = std::chrono::duration_cast<sec>(end - start).count();
                std::cout << "kernel duration=" << duration << "s" << std::endl;
#endif
            }

            // cleanup
            CHECK_CUDART(cudaStreamSynchronize(ctx->stream)); //needed until cuda11 support is there
            CHECK_CUDART(cudaFree(d_bins));
            CHECK_CUDART(cudaFree(d_bin_sizes));
        }
        else {
            CHECK_CUDA(op->program->kernel(op_name)
                    .instantiate(type_of(value_type), std::max(static_cast<uint32_t>(1), dbw->num_sides()),
                            op->num_temp_vectors, tmp_len)
                    .configure(grid, block, shared_mem_size, ctx->stream)
                    .launch(dbw->d_in<T>(0), dbw->d_sides<T>(), dbw->d_out<T>(), dbw->d_scalars<T>(), d_temp,
                            dbw->grix(), nullptr, 0, 0));
            
        }
		if(op->num_temp_vectors > 0) {
            CHECK_CUDART(cudaStreamSynchronize(ctx->stream));
            CHECK_CUDART(cudaFree(d_temp));
        }
	}
};
