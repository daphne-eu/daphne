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

class WorkerImpl final : public distributed::Worker::AsyncService
{
public:
    const static std::string DISTRIBUTED_FUNCTION_NAME;
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;

    WorkerImpl();
    ~WorkerImpl() override;

    void HandleRpcs();
    // void StartHandleThread();
    // void TerminateHandleThread();
    grpc::Status Store(::grpc::ServerContext *context,
                         const ::distributed::Matrix *request,
                         ::distributed::StoredData *response) override;
    grpc::Status Compute(::grpc::ServerContext *context,
                         const ::distributed::Task *request,
                         ::distributed::ComputeResult *response) override;
    grpc::Status Transfer(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                          ::distributed::Matrix *response) override;
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

    DenseMatrix<double> *readOrGetMatrix(const std::string &filename, size_t numRows, size_t numCols);
    void *loadWorkInputData(mlir::Type mlirType, const distributed::WorkData& workInput);
    static distributed::WorkData::DataCase dataCaseForType(mlir::Type type) ;    

  class CallData{
      public:
        virtual void Proceed() = 0;
  };
  class StoreCallData final : public CallData{
   public:
    StoreCallData(distributed::Worker::AsyncService * service, grpc::ServerCompletionQueue* cq)
        : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
      // Invoke the serving logic right away.
      Proceed();
    }

    void Proceed() {
      if (status_ == CREATE) {
        // Make this instance progress to the PROCESS state.
        status_ = PROCESS;

        service_->RequestStore (&ctx_, &matrix, &responder_, cq_, cq_,
                                  this);
      } else if (status_ == PROCESS) {
        status_ = FINISH;

        new StoreCallData(service_, cq_);
        grpc::Status status = service_->Store(&ctx_, &matrix, &storedData);
        
        responder_.Finish(storedData, grpc::Status::OK, this);
      } else {
        GPR_ASSERT(status_ == FINISH);        
        delete this;
      }
    }

   private:
    distributed::Worker::AsyncService* service_;
    // The producer-consumer queue where for asynchronous server notifications.
    grpc::ServerCompletionQueue* cq_;        
    grpc::ServerContext ctx_;
    // What we get from the client.
    distributed::Matrix matrix;
    // What we send back to the client.
    distributed::StoredData storedData;    
    // The means to get back to the client.
    grpc::ServerAsyncResponseWriter<distributed::StoredData> responder_;

    // Let's implement a tiny state machine with the following states.
    enum CallStatus { CREATE, PROCESS, FINISH };    
    CallStatus status_;  // The current serving state.
  };
    class ComputeCallData final : public CallData{
    public:
        ComputeCallData(distributed::Worker::AsyncService * service, grpc::ServerCompletionQueue* cq)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
        // Invoke the serving logic right away.
        Proceed();
        }

        void Proceed() {
        if (status_ == CREATE) {
            // Make this instance progress to the PROCESS state.
            status_ = PROCESS;

            service_->RequestCompute (&ctx_, &task, &responder_, cq_, cq_,
                                    this);
        } else if (status_ == PROCESS) {
            status_ = FINISH;

            new ComputeCallData(service_, cq_);

            grpc::Status status = service_->Compute(&ctx_, &task, &result);
            
            responder_.Finish(result, status, this);
        } else {
            GPR_ASSERT(status_ == FINISH);        
            delete this;
        }
        }

    private:
        distributed::Worker::AsyncService* service_;
        // The producer-consumer queue where for asynchronous server notifications.
        grpc::ServerCompletionQueue* cq_;        
        grpc::ServerContext ctx_;
        // What we get from the client.
        distributed::Task task;
        // What we send back to the client.
        distributed::ComputeResult result;    
        // The means to get back to the client.
        grpc::ServerAsyncResponseWriter<distributed::ComputeResult> responder_;

        // Let's implement a tiny state machine with the following states.
        enum CallStatus { CREATE, PROCESS, FINISH };    
        CallStatus status_;  // The current serving state.
    };


    class TransferCallData final : public CallData{
    public:
        TransferCallData(distributed::Worker::AsyncService * service, grpc::ServerCompletionQueue* cq)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
        // Invoke the serving logic right away.
        Proceed();
        }

        void Proceed() {
        if (status_ == CREATE) {
            // Make this instance progress to the PROCESS state.
            status_ = PROCESS;

            service_->RequestTransfer (&ctx_, &storedData, &responder_, cq_, cq_,
                                    this);
        } else if (status_ == PROCESS) {
            status_ = FINISH;
        
            new TransferCallData(service_, cq_);

            grpc::Status status = service_->Transfer(&ctx_, &storedData, &matrix);
            
            responder_.Finish(matrix, status, this);
        } else {
            GPR_ASSERT(status_ == FINISH);        
            delete this;
        }
        }

    private:
        distributed::Worker::AsyncService* service_;
        // The producer-consumer queue where for asynchronous server notifications.
        grpc::ServerCompletionQueue* cq_;        
        grpc::ServerContext ctx_;
        // What we get from the client.
        distributed::StoredData storedData;    
        // What we send back to the client.
        distributed::Matrix matrix;
        // The means to get back to the client.
        grpc::ServerAsyncResponseWriter<distributed::Matrix> responder_;

        // Let's implement a tiny state machine with the following states.
        enum CallStatus { CREATE, PROCESS, FINISH };    
        CallStatus status_;  // The current serving state.
    };
};

#endif //SRC_RUNTIME_DISTRIBUTED_WORKER_WORKERIMPL_H
