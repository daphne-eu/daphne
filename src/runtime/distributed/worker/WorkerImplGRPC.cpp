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


#include "WorkerImplGRPC.h"

#include <runtime/distributed/proto/CallData.h>
#include <runtime/distributed/proto/ProtoDataConverter.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

WorkerImplGRPC::WorkerImplGRPC(std::string addr)
{
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    cq_ = builder.AddCompletionQueue();
    builder.RegisterService(&service_);
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    server = builder.BuildAndStart();
}

void WorkerImplGRPC::Wait() {
    // Spawn a new CallData instance to serve new clients.
    new StoreCallData(this, cq_.get());
    new ComputeCallData(this, cq_.get());
    new TransferCallData(this, cq_.get());
    // new FreeMemCallData(this, cq_.get());
    void* tag;  // uniquely identifies a request.
    bool ok;
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    // The return value of Next should always be checked. This return value
    // tells us whether there is any kind of event or cq_ is shutting down.
    while (cq_->Next(&tag, &ok)) {        
        if(ok){         
            // Thread pool ? with caution. For now on each worker only one thread operates (sefe IO).
            // We might need to add locks inside Store/Compute/Transfer methods if we deploy threads
            static_cast<CallData*>(tag)->Proceed();
        } else {
            // TODO maybe handle this internally ?
            delete static_cast<CallData*>(tag);
        }
    }
}
template<>
DenseMatrix<double>* WorkerImplGRPC::CreateMatrix<DenseMatrix<double>>(const ::distributed::Matrix *mat) {
    return DataObjectFactory::create<DenseMatrix<double>>(mat->num_rows(), mat->num_cols(), false);
}
template<>
DenseMatrix<int64_t>* WorkerImplGRPC::CreateMatrix<DenseMatrix<int64_t>>(const ::distributed::Matrix *mat) {
    return DataObjectFactory::create<DenseMatrix<int64_t>>(mat->num_rows(), mat->num_cols(), false);
}
template<>
CSRMatrix<double>* WorkerImplGRPC::CreateMatrix<CSRMatrix<double>>(const ::distributed::Matrix *mat) {
    return DataObjectFactory::create<CSRMatrix<double>>(mat->num_rows(), mat->num_cols(), mat->csr_matrix().values_f64().cells_size(), true);
}
template<>
CSRMatrix<int64_t>* WorkerImplGRPC::CreateMatrix<CSRMatrix<int64_t>>(const ::distributed::Matrix *mat) {
    return DataObjectFactory::create<CSRMatrix<int64_t>>(mat->num_rows(), mat->num_cols(), mat->csr_matrix().values_i64().cells_size(), true);
}

grpc::Status WorkerImplGRPC::StoreGRPC(::grpc::ServerContext *context,
                         const ::distributed::Matrix *request,
                         ::distributed::StoredData *response) 
{
    Structure *mat;
    switch (request->matrix_case()){
    case distributed::Matrix::MatrixCase::kDenseMatrix:
        switch (request->dense_matrix().cells_case())
        {
        case distributed::DenseMatrix::CellsCase::kCellsI64:
            mat = CreateMatrix<DenseMatrix<int64_t>>(request);
            ProtoDataConverter<DenseMatrix<int64_t>>::convertFromProto(*request, dynamic_cast<DenseMatrix<int64_t>*>(mat));
            break;
        case distributed::DenseMatrix::CellsCase::kCellsF64:
            mat = CreateMatrix<DenseMatrix<double>>(request);
            ProtoDataConverter<DenseMatrix<double>>::convertFromProto(*request, dynamic_cast<DenseMatrix<double>*>(mat));
            break;    
        default:
            break;
        }
        break;
    case distributed::Matrix::MatrixCase::kCsrMatrix:
        switch (request->csr_matrix().values_case())
        {
        case distributed::CSRMatrix::ValuesCase::kValuesI64:
            mat = CreateMatrix<CSRMatrix<int64_t>>(request);
            ProtoDataConverter<CSRMatrix<int64_t>>::convertFromProto(*request, dynamic_cast<CSRMatrix<int64_t>*>(mat));
            break;
        case distributed::CSRMatrix::ValuesCase::kValuesF64:
            mat = CreateMatrix<CSRMatrix<double>>(request);
            ProtoDataConverter<CSRMatrix<double>>::convertFromProto(*request, dynamic_cast<CSRMatrix<double>*>(mat));
            break;
        }
        
    default:
        // error message         
        return ::grpc::Status::CANCELLED;
        break;
    };
    auto storedInfo = Store(mat);

    response->set_filename(storedInfo.filename);
    response->set_num_rows(storedInfo.numRows);
    response->set_num_cols(storedInfo.numCols);
    return ::grpc::Status::OK;
}

grpc::Status WorkerImplGRPC::ComputeGRPC(::grpc::ServerContext *context,
                         const ::distributed::Task *request,
                         ::distributed::ComputeResult *response)
{
    std::vector<StoredInfo> inputs;
    inputs.reserve(request->inputs().size());

    std::vector<StoredInfo> *outputs = new std::vector<StoredInfo>();
    for (auto input : request->inputs()){
        auto stored = input.stored();
        inputs.push_back(StoredInfo({stored.filename(), stored.num_rows(), stored.num_cols()}));
    }
    auto respMsg = Compute(outputs, inputs, request->mlir_code());
    for (auto output : *outputs){        
        distributed::WorkData workData;        
        workData.mutable_stored()->set_filename(output.filename);
        workData.mutable_stored()->set_num_rows(output.numRows);
        workData.mutable_stored()->set_num_cols(output.numCols);
        *response->add_outputs() = workData;
    }
    if (respMsg == std::string("OK"))
        return ::grpc::Status::OK;
    else
        return ::grpc::Status(grpc::StatusCode::ABORTED, respMsg);
}

grpc::Status WorkerImplGRPC::TransferGRPC(::grpc::ServerContext *context,
                          const ::distributed::StoredData *request,
                         ::distributed::Matrix *response)
{
    StoredInfo info({request->filename(), request->num_rows(), request->num_cols()});
    Structure *mat = Transfer(info);
    if(auto matDT = dynamic_cast<DenseMatrix<double>*>(mat))        
        ProtoDataConverter<DenseMatrix<double>>::convertToProto(matDT, response);
    else if(auto matDT = dynamic_cast<DenseMatrix<int64_t>*>(mat))        
        ProtoDataConverter<DenseMatrix<int64_t>>::convertToProto(matDT, response);
    else if(auto matDT = dynamic_cast<CSRMatrix<int64_t>*>(mat))        
        ProtoDataConverter<CSRMatrix<int64_t>>::convertToProto(matDT, response);
    else if(auto matDT = dynamic_cast<CSRMatrix<double>*>(mat))        
        ProtoDataConverter<CSRMatrix<double>>::convertToProto(matDT, response);
    else 
        std::runtime_error("Type is not supported atm");
    return ::grpc::Status::OK;
}