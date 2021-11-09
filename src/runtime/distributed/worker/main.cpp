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

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

#include <iostream>

#include "WorkerImpl.h"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <Address:Port>" << std::endl;
        exit(1);
    }
    auto addr = argv[1];
    
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    
    WorkerImpl my_service;

    my_service.cq_ = builder.AddCompletionQueue();
    builder.RegisterService(&my_service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    std::cout << "Started Distributed Worker on `" << addr << "`\n";
    my_service.HandleRpcs();
    // TODO shutdown handling
    // server->Shutdown();
    // my_service.cq_->Shutdown();

    return 0;
}