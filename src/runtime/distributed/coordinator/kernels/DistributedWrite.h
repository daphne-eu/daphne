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

#pragma once

#include <runtime/local/context/DistributedContext.h>
#include <runtime/local/context/HDFSContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/worker/WorkerImpl.h>
#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>

#ifdef USE_MPI
#include <runtime/distributed/worker/MPIHelper.h>
#endif

#include <cstddef>
#include <filesystem>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <ALLOCATION_TYPE AT, class DTArg> struct DistributedWrite {
    static void apply(const DTArg *mat, const char *filename, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> void distributedWrite(const DTArg *mat, const char *filename, DCTX(dctx)) {
    const auto allocation_type = dctx->getUserConfig().distributedBackEndSetup;
    if (allocation_type == ALLOCATION_TYPE::DIST_MPI) {
#ifdef USE_MPI
        DistributedWrite<ALLOCATION_TYPE::DIST_MPI, const DTArg>::apply(mat, filename, dctx);
#endif
    } else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_ASYNC) {
        DistributedWrite<ALLOCATION_TYPE::DIST_GRPC_ASYNC, const DTArg>::apply(mat, filename, dctx);
    } else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_SYNC) {
        DistributedWrite<ALLOCATION_TYPE::DIST_GRPC_SYNC, const DTArg>::apply(mat, filename, dctx);
    }
}

// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

#ifdef USE_MPI
// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
template <class DTArg> struct DistributedWrite<ALLOCATION_TYPE::DIST_MPI, DTArg> {
    static void apply(const DTArg *mat, const char *filename, DCTX(dctx)) {
        throw std::runtime_error("not implemented");
    }
};
#endif

// ----------------------------------------------------------------------------
// Asynchronous GRPC
// ----------------------------------------------------------------------------

template <class DTArg> struct DistributedWrite<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DTArg> {
    static void apply(const DTArg *mat, const char *filename, DCTX(dctx)) {
        throw std::runtime_error("not implemented");
    }
};

// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------
#ifdef USE_HDFS

template <class DTArg> struct DistributedWrite<ALLOCATION_TYPE::DIST_GRPC_SYNC, DTArg> {
    static void apply(const DTArg *mat, const char *filename, DCTX(dctx)) {
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();

        if (mat == nullptr) {
            throw std::runtime_error("matrix argument is null");
        }

        std::filesystem::path filePath(filename);
        auto directoryName = "/" + filePath.filename().string();
        auto baseFileName = directoryName + "/" + filePath.filename().string();

        // Get nested file extension
        auto extension = filePath.stem().extension().string();

        auto hdfsCtx = HDFSContext::get(dctx);
        auto fs = hdfsCtx->getConnection();
        if (fs == NULL) {
            std::cout << "Error connecting to HDFS" << std::endl;
        }
        // Initialize chunk directory
        if (hdfsCreateDirectory(*fs, directoryName.c_str())) {
            throw std::runtime_error("Directory failed");
        }
        size_t chunkId = 1;
        std::vector<std::thread> threads_vector;
        for (auto workerAddr : workers) {
            auto hdfsfilename = baseFileName + "_segment_" + std::to_string(chunkId++);
            DataPlacement *dp;
            if ((dp = mat->getMetaDataObject()->getDataPlacementByLocation(workerAddr))) {
                if (auto grpc_alloc = dynamic_cast<AllocationDescriptorGRPC *>(dp->getAllocation(0))) {
                    auto data = grpc_alloc->getDistributedData();

                    if (data.isPlacedAtWorker) {
                        std::thread t([=, &mat]() {
                            auto stub = ctx->stubs[workerAddr].get();

                            distributed::HDFSWriteInfo fileData;
                            fileData.mutable_matrix()->set_identifier(data.identifier);
                            fileData.mutable_matrix()->set_num_rows(data.numRows);
                            fileData.mutable_matrix()->set_num_cols(data.numCols);

                            fileData.set_dirname(hdfsfilename.c_str());
                            fileData.set_segment(std::to_string(chunkId).c_str());

                            grpc::ClientContext grpc_ctx;
                            distributed::Empty empty;

                            auto status = stub->WriteHDFS(&grpc_ctx, fileData, &empty);
                            if (!status.ok())
                                throw std::runtime_error(status.error_message());
                        });
                        threads_vector.push_back(move(t));
                    } else {
                        auto slicedMat =
                            mat->sliceRow(dp->getRange()->r_start, dp->getRange()->r_start + dp->getRange()->r_len);
                        if (extension == ".csv") {
                            writeHDFSCsv(slicedMat, hdfsfilename.c_str(), dctx);
                        } else if (extension == ".dbdf") {
                            writeDaphneHDFS(slicedMat, hdfsfilename.c_str(), dctx);
                        }
                    }
                } else {
                    continue;
                }
                // TODO we should also store ranges that did not have a
                // dataplacement associated with them
            } else
                throw std::runtime_error("dynamic_cast<AllocationDescriptorGRPC*>(alloc) failed (returned nullptr)");
        }
        for (auto &thread : threads_vector)
            thread.join();
    }
};
#endif