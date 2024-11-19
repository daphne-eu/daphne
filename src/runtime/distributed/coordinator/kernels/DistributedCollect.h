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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/context/DistributedContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinaryMat.h>

#include <runtime/distributed/proto/DistributedGRPCCaller.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>
#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/local/datastructures/AllocationDescriptorGRPC.h>
#include <runtime/local/io/DaphneSerializer.h>

#ifdef USE_MPI
#include <runtime/distributed/worker/MPIHelper.h>
#endif

#include <cstddef>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <ALLOCATION_TYPE AT, class DT> struct DistributedCollect {
    static void apply(DT *&mat, const VectorCombine &combine, DCTX(dctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <ALLOCATION_TYPE AT, class DT> void distributedCollect(DT *&mat, const VectorCombine &combine, DCTX(dctx)) {
    DistributedCollect<AT, DT>::apply(mat, combine, dctx);
}

// ****************************************************************************
// (Partial) template specializations for different distributed backends
// ****************************************************************************

// ----------------------------------------------------------------------------
// MPI
// ----------------------------------------------------------------------------
#ifdef USE_MPI
template <class DT> struct DistributedCollect<ALLOCATION_TYPE::DIST_MPI, DT> {

    static void apply(DT *&mat, const VectorCombine &combine, DCTX(dctx)) {
        if (mat == nullptr)
            throw std::runtime_error("DistributedCollect gRPC: result matrix must be already "
                                     "allocated by wrapper since information regarding size only "
                                     "exists there");
        size_t worldSize = MPIHelper::getCommSize();
        for (size_t rank = 0; rank < worldSize; rank++) {
            if (rank == COORDINATOR) // we currently exclude the coordinator
                continue;

            std::string address = std::to_string(rank);
            auto dp = mat->getMetaDataObject()->getDataPlacementByLocation(address);
            auto distributedData = dynamic_cast<AllocationDescriptorMPI *>(dp->getAllocation(0))->getDistributedData();
            WorkerImpl::StoredInfo info = {distributedData.identifier, distributedData.numRows,
                                           distributedData.numCols};
            MPIHelper::requestData(rank, info);
        }
        auto collectedDataItems = 0u;
        for (size_t i = 1; i < worldSize; i++) {
            size_t len;
            int rank;
            std::vector<char> buffer;
            MPIHelper::getMessage(&rank, TypesOfMessages::OUTPUT, MPI_UNSIGNED_CHAR, buffer, &len);

            std::string address = std::to_string(rank);
            auto dp = mat->getMetaDataObject()->getDataPlacementByLocation(address);

            auto denseMat = dynamic_cast<DenseMatrix<double> *>(mat);
            if (!denseMat) {
                throw std::runtime_error("Distribute grpc only supports "
                                         "DenseMatrix<double> for now");
            }

            auto slicedMat = dynamic_cast<DenseMatrix<double> *>(DF_deserialize(buffer));
            if (combine == VectorCombine::ADD) {
                ewBinaryMat(BinaryOpCode::ADD, denseMat, slicedMat, denseMat, nullptr);
            } else {
                auto resValues = denseMat->getValues() + (dp->getRange()->r_start * denseMat->getRowSkip());
                auto slicedMatValues = slicedMat->getValues();
                for (size_t r = 0; r < dp->getRange()->r_len; r++) {
                    memcpy(resValues + dp->getRange()->c_start, slicedMatValues,
                           dp->getRange()->c_len * sizeof(double));
                    resValues += denseMat->getRowSkip();
                    slicedMatValues += slicedMat->getRowSkip();
                }
            }
            DataObjectFactory::destroy(slicedMat);

            collectedDataItems += dp->getRange()->r_len * dp->getRange()->c_len;

            auto distributedData = dynamic_cast<AllocationDescriptorMPI *>(dp->getAllocation(0))->getDistributedData();
            distributedData.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorMPI *>(dp->getAllocation(0))->updateDistributedData(distributedData);
            // this is to handle the case when not all workers participate in
            // the computation, i.e., number of workers is larger than of the
            // work items
            if (collectedDataItems == denseMat->getNumRows() * denseMat->getNumCols())
                break;
        }
    };
};
#endif

// ----------------------------------------------------------------------------
// Asynchronous GRPC
// ----------------------------------------------------------------------------

template <class DT> struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_ASYNC, DT> {

    static void apply(DT *&mat, const VectorCombine &combine, DCTX(dctx)) {
        if (mat == nullptr)
            throw std::runtime_error("DistributedCollect gRPC: result matrix must be already "
                                     "allocated by wrapper since information regarding size only "
                                     "exists there");

        struct StoredInfo {
            size_t dp_id;
        };
        DistributedGRPCCaller<StoredInfo, distributed::StoredData, distributed::Data> caller(dctx);

        auto dpVector = mat->getMetaDataObject()->getRangeDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->getAllocation(0)->getLocation();

            auto distributedData = dynamic_cast<AllocationDescriptorGRPC *>(dp->getAllocation(0))->getDistributedData();
            StoredInfo storedInfo({dp->getID()});
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);

            caller.asyncTransferCall(address, storedInfo, protoData);
        }

        while (!caller.isQueueEmpty()) {
            auto response = caller.getNextResult();
            auto dp_id = response.storedInfo.dp_id;
            auto dp = mat->getMetaDataObject()->getDataPlacementByID(dp_id);
            auto data = dynamic_cast<AllocationDescriptorGRPC *>(dp->getAllocation(0))->getDistributedData();

            auto matProto = response.result;

            // TODO: We need to handle different data types
            auto denseMat = dynamic_cast<DenseMatrix<double> *>(mat);
            if (!denseMat) {
                throw std::runtime_error("Distribute grpc only supports "
                                         "DenseMatrix<double> for now");
            }
            // Zero copy buffer
            std::vector<char> buf(static_cast<const char *>(matProto.bytes().data()),
                                  static_cast<const char *>(matProto.bytes().data()) + matProto.bytes().size());
            auto slicedMat = dynamic_cast<DenseMatrix<double> *>(DF_deserialize(buf));
            if (combine == VectorCombine::ADD) {
                ewBinaryMat(BinaryOpCode::ADD, denseMat, slicedMat, denseMat, nullptr);
            } else {
                auto resValues = denseMat->getValues() + (dp->getRange()->r_start * denseMat->getRowSkip());
                auto slicedMatValues = slicedMat->getValues();
                for (size_t r = 0; r < dp->getRange()->r_len; r++) {
                    memcpy(resValues + dp->getRange()->c_start, slicedMatValues,
                           dp->getRange()->c_len * sizeof(double));
                    resValues += denseMat->getRowSkip();
                    slicedMatValues += slicedMat->getRowSkip();
                }
            }
            DataObjectFactory::destroy(slicedMat);

            data.isPlacedAtWorker = false;
            dynamic_cast<AllocationDescriptorGRPC *>(dp->getAllocation(0))->updateDistributedData(data);
        }
    };
};

// ----------------------------------------------------------------------------
// Synchronous GRPC
// ----------------------------------------------------------------------------

template <class DT> struct DistributedCollect<ALLOCATION_TYPE::DIST_GRPC_SYNC, DT> {

    static void apply(DT *&mat, const VectorCombine &combine, DCTX(dctx)) {
        if (mat == nullptr)
            throw std::runtime_error("DistributedCollect gRPC: result matrix must be already "
                                     "allocated by wrapper since information regarding size only "
                                     "exists there");

        auto ctx = DistributedContext::get(dctx);
        std::vector<std::thread> threads_vector;
        std::mutex lock;

        auto dpVector = mat->getMetaDataObject()->getRangeDataPlacementByType(ALLOCATION_TYPE::DIST_GRPC);
        for (auto &dp : *dpVector) {
            auto address = dp->getAllocation(0)->getLocation();

            auto distributedData = dynamic_cast<AllocationDescriptorGRPC *>(dp->getAllocation(0))->getDistributedData();
            distributed::StoredData protoData;
            protoData.set_identifier(distributedData.identifier);
            protoData.set_num_rows(distributedData.numRows);
            protoData.set_num_cols(distributedData.numCols);

            std::thread t([address, dp = dp.get(), protoData, distributedData, &combine, &lock, &mat, &ctx]() mutable {
                auto stub = ctx->stubs[address].get();

                distributed::Data matProto;
                grpc::ClientContext grpc_ctx;
                stub->Transfer(&grpc_ctx, protoData, &matProto);

                // TODO: We need to handle different data types
                auto denseMat = dynamic_cast<DenseMatrix<double> *>(mat);
                if (!denseMat) {
                    throw std::runtime_error("Distribute grpc only supports "
                                             "DenseMatrix<double> for now");
                }
                // Zero copy buffer
                std::vector<char> buf(static_cast<const char *>(matProto.bytes().data()),
                                      static_cast<const char *>(matProto.bytes().data()) + matProto.bytes().size());
                auto slicedMat = dynamic_cast<DenseMatrix<double> *>(DF_deserialize(buf));
                if (combine == VectorCombine::ADD) {
                    std::lock_guard g(lock);
                    ewBinaryMat(BinaryOpCode::ADD, denseMat, slicedMat, denseMat, nullptr);
                } else {
                    auto resValues = denseMat->getValues() + (dp->getRange()->r_start * denseMat->getRowSkip());
                    auto slicedMatValues = slicedMat->getValues();
                    for (size_t r = 0; r < dp->getRange()->r_len; r++) {
                        memcpy(resValues + dp->getRange()->c_start, slicedMatValues,
                               dp->getRange()->c_len * sizeof(double));
                        resValues += denseMat->getRowSkip();
                        slicedMatValues += slicedMat->getRowSkip();
                    }
                }
                DataObjectFactory::destroy(slicedMat);
                distributedData.isPlacedAtWorker = false;
                dynamic_cast<AllocationDescriptorGRPC *>(dp->getAllocation(0))->updateDistributedData(distributedData);
            });
            threads_vector.push_back(std::move(t));
        }
        for (auto &thread : threads_vector)
            thread.join();
    };
};
