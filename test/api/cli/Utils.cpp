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

#include <api/cli/Utils.h>
#include <runtime/distributed/worker/WorkerImpl.h>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

std::string readTextFile(const std::string & filePath) {
    std::ifstream ifs(filePath, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("could not open file '" + filePath + "'");
    
    std::stringstream stream;
    stream << ifs.rdbuf();
    
    return stream.str();
}

[[maybe_unused]] std::unique_ptr<grpc::Server> startDistributedWorker(const char *addr, WorkerImpl *workerImpl)
{
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    workerImpl->cq_ = builder.AddCompletionQueue();
    builder.RegisterService(&workerImpl->service_);    
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    
    return builder.BuildAndStart();
}

std::string generalizeDataTypes(const std::string& str) {
    std::regex re("(DenseMatrix|CSRMatrix)");
    return std::regex_replace(str, re, "<SomeMatrix>");
}