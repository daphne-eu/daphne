/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H
#define SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H

#include <map>

#include <mlir/IR/BuiltinTypes.h>

#include <runtime/local/datastructures/DenseMatrix.h>
#include "runtime/distributed/proto/worker.pb.h"
#include "runtime/distributed/proto/worker.grpc.pb.h"

class WorkerImpl final 
{
public:
    const static std::string DISTRIBUTED_FUNCTION_NAME;
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;

    WorkerImpl();
    ~WorkerImpl();
    
    void HandleRpcs();
    // void StartHandleThread();
    // void TerminateHandleThread();
    grpc::Status Store(::grpc::ServerContext *context,
                         const ::distributed::Matrix *request,
                         ::distributed::StoredData *response) ;
    grpc::Status Compute(::grpc::ServerContext *context,
                         const ::distributed::Task *request,
                         ::distributed::ComputeResult *response) ;
    grpc::Status Transfer(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Matrix *response);
    grpc::Status FreeMem(::grpc::ServerContext *context,
                         const ::distributed::StoredData *request,
                         ::distributed::Empty *emptyMessage);
    distributed::Worker::AsyncService service_;
private:
    uint64_t tmp_file_counter_ = 0;
    std::unordered_map<std::string, void *> localData_;
    /**
     * Creates a vector holding pointers to the inputs as well as the outputs. This vector can directly be passed
     * to the `ExecutionEngine::invokePacked` method.
     * @param functionType Type of the function that will be invoked
     * @param workInputs Inputs send by client
     * @param outputs Reference to the vector that will hold the outputs of the invoked function
     * @return packed pointers to inputs and outputs
     */
    std::vector<void *> createPackedCInterfaceInputsOutputs(mlir::FunctionType functionType,
                                                            google::protobuf::RepeatedPtrField<distributed::WorkData> workInputs,
                                                            std::vector<void *> &outputs,
                                                            std::vector<void *> &inputs);
    
    template<typename VT>
    Matrix<VT> *readOrGetMatrix(const std::string &filename, size_t numRows, size_t numCols, bool isSparse);
    void *loadWorkInputData(mlir::Type mlirType, const distributed::WorkData& workInput);
    static distributed::WorkData::DataCase dataCaseForType(mlir::Type type);
    
    /**
     * @brief Helper store function using templates in order to handle
     * different types.
     * 
     * @tparam VT double/int etc.
     */
    template<typename VT>
    grpc::Status templateStore(::grpc::ServerContext *context,
                         const ::distributed::Matrix *request,
                         ::distributed::StoredData *response) ;

    /**
     * @brief Helper transfer function using templates in order to handle
     * different types.
     * 
     * @tparam VT double/int etc.
     */
    template<typename VT>
    grpc::Status templateTransfer(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Matrix *response);
                         /**
     * @brief Helper FreeMem function using templates in order to handle
     * different types.
     * 
     * @tparam VT double/int etc.
     */
    template<typename VT>
    grpc::Status templateFreeMem(::grpc::ServerContext *context,
                         const ::distributed::StoredData *request,
                         ::distributed::Empty *emptyMessage);
};

#endif //SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H
