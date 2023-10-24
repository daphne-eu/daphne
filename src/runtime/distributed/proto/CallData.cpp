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

#include "CallData.h"

void StoreCallData::Proceed(bool ok) {
    if (status_ == CREATE)
    {
        // Make this instance progress to the PROCESS state.
        status_ = PROCESS;

        service_->RequestStore(&ctx_, &stream_, cq_, cq_, this);
        worker->PrepareStoreGRPC();
    }
    else if (status_ == PROCESS)
    {
        if (ok) {
            stream_.Read(&data, this);
            grpc::Status status = worker->StoreGRPC(&ctx_, &data, &storedData);
            if (!status.ok())
                throw std::runtime_error("Error while receiving/storing partial data");            
        }
        else {
            new StoreCallData(worker, scq_, cq_);
            status_ = FINISH;

            stream_.Finish(storedData, grpc::Status::OK, this);
        }
    }
    else
    {
        GPR_ASSERT(status_ == FINISH);
        delete this;
    }
}

void ComputeCallData::Proceed(bool ok) {
    if (status_ == CREATE)
    {
        // Make this instance progress to the PROCESS state.
        status_ = PROCESS;

        service_->RequestCompute(&ctx_, &task, &responder_, cq_, cq_,
                                    this);
    }
    else if (status_ == PROCESS)
    {
        if (!ok)
            delete this;
        status_ = FINISH;

        new ComputeCallData(worker, cq_);

        grpc::Status status = worker->ComputeGRPC(&ctx_, &task, &result);

        responder_.Finish(result, status, this);
    }
    else
    {
        GPR_ASSERT(status_ == FINISH);
        delete this;
    }
}

void TransferCallData::Proceed(bool ok) {
    if (status_ == CREATE)
    {
        // Make this instance progress to the PROCESS state.
        status_ = PROCESS;

        service_->RequestTransfer(&ctx_, &storedData, &responder_, cq_, cq_,
                                    this);
    }
    else if (status_ == PROCESS)
    {
        if (!ok)
            delete this;
        status_ = FINISH;

        new TransferCallData(worker, cq_);

        grpc::Status status = worker->TransferGRPC(&ctx_, &storedData, &data);

        responder_.Finish(data, status, this);
    }
    else
    {
        GPR_ASSERT(status_ == FINISH);
        delete this;
    }
}


// void FreeMemCallData::Proceed() {
//     if (status_ == CREATE)
//     {
//         // Make this instance progress to the PROCESS state.
//         status_ = PROCESS;

//         service_->RequestFreeMem(&ctx_, &storedData, &responder_, cq_, cq_,
//                                     this);
//     }
//     else if (status_ == PROCESS)
//     {
//         status_ = FINISH;

//         new FreeMemCallData(worker, cq_);

//         grpc::Status status = worker->FreeMem(&ctx_, &storedData, &emptyMessage);

//         responder_.Finish(emptyMessage, status, this);
//     }
//     else
//     {
//         GPR_ASSERT(status_ == FINISH);
//         delete this;
//     }
// }