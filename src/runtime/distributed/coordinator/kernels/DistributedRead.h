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

#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/context/DistributedContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>

#include <runtime/distributed/coordinator/scheduling/LoadPartitioningDistributed.h>
#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/worker/WorkerImpl.h>
#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>

#ifdef USE_MPI
#include <runtime/distributed/worker/MPIHelper.h>
#endif

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <ALLOCATION_TYPE AT, class DTRes> struct DistributedRead {
    static void apply(DTRes *&res, const char *filename, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void distributedRead(DTRes *&res, const char *filename, DCTX(dctx)) {
    const auto allocation_type = dctx->getUserConfig().distributedBackEndSetup;
    if (allocation_type == ALLOCATION_TYPE::DIST_MPI) {
#ifdef USE_MPI
        DistributedRead<ALLOCATION_TYPE::DIST_MPI, DTRes>::apply(res, filename,
                                                                 dctx);
#endif
    } else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_ASYNC) {
        DistributedRead<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DTRes>::apply(
            res, filename, dctx);
    } else if (allocation_type == ALLOCATION_TYPE::DIST_GRPC_SYNC) {
        DistributedRead<ALLOCATION_TYPE::DIST_GRPC_SYNC, DTRes>::apply(
            res, filename, dctx);
    }
}

// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

#ifdef USE_MPI
// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
template <class DTRes>
struct DistributedRead<ALLOCATION_TYPE::DIST_MPI, DTRes> {
    static void apply(DTRes *&res, const char *filename, DCTX(dctx)) {
        throw std::runtime_error("not implemented");
    }
};
#endif

// ----------------------------------------------------------------------------
// Asynchronous GRPC
// ----------------------------------------------------------------------------

template <class DTRes>
struct DistributedRead<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DTRes> {
    static void apply(DTRes *&res, const char *filename, DCTX(dctx)) {
        throw std::runtime_error("not implemented");
    }
};

// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------

template <class DTRes>
struct DistributedRead<ALLOCATION_TYPE::DIST_GRPC_SYNC, DTRes> {
    static void apply(DTRes *&res, const char *filename, DCTX(dctx)) {
#if USE_HDFS
        auto ctx = DistributedContext::get(dctx);
        auto workers = ctx->getWorkers();

        // Generate metadata for the object based on MetaDataFile and
        // when the worker needs the data it will read it automatically

        std::vector<std::thread> threads_vector;
        LoadPartitioningDistributed<DTRes, AllocationDescriptorGRPC> partioner(
            DistributionSchema::DISTRIBUTE, res, dctx);
        while (partioner.HasNextChunk()) {
            auto hdfsFn = std::string(filename);
            auto dp = partioner.GetNextChunk();

            auto workerAddr =
                dynamic_cast<AllocationDescriptorGRPC *>(dp->allocation.get())
                    ->getLocation();
            std::thread t([=, &res]() {
                auto stub = ctx->stubs[workerAddr].get();

                distributed::HDFSFile fileData;
                fileData.set_filename(hdfsFn);
                fileData.set_start_row(dp->range->r_start);
                fileData.set_num_rows(dp->range->r_len);
                fileData.set_num_cols(dp->range->c_len);

                grpc::ClientContext grpc_ctx;
                distributed::StoredData response;

                auto status = stub->ReadHDFS(&grpc_ctx, fileData, &response);
                if (!status.ok())
                    throw std::runtime_error(status.error_message());

                DistributedData newData;
                newData.identifier = response.identifier();
                newData.numRows = response.num_rows();
                newData.numCols = response.num_cols();
                newData.isPlacedAtWorker = true;
                dynamic_cast<AllocationDescriptorGRPC &>(*(dp->allocation))
                    .updateDistributedData(newData);
            });
            threads_vector.push_back(move(t));
        }

        for (auto &thread : threads_vector)
            thread.join();
#endif
    }
};
