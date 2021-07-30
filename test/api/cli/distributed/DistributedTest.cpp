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

#include <runtime/distributed/worker/WorkerImpl.h>
#include <api/cli/Utils.h>

#include <cstdlib>

#include <tags.h>

#include <catch.hpp>

#include <grpcpp/grpcpp.h>

const std::string dirPath = "test/api/cli/distributed/";

TEST_CASE("Simple distributed execution test", TAG_DISTRIBUTED)
{
    auto addr1 = "0.0.0.0:50051";
    auto addr2 = "0.0.0.0:50052";
    WorkerImpl workerImpl1;
    WorkerImpl workerImpl2;
    auto server1 = startDistributedWorker(addr1, &workerImpl1);
    auto server2 = startDistributedWorker(addr2, &workerImpl2);
    auto distWorkerStr = std::string(addr1) + ',' + addr2;

    assert(std::getenv("DISTRIBUTED_WORKERS") == nullptr);
    for (auto i = 1u; i < 3; ++i) {
        auto filename = dirPath + "distributed_" + std::to_string(i) + ".daphne";

        std::stringstream outLocal;
        std::stringstream errLocal;
        int status = runDaphne(outLocal, errLocal, filename.c_str());

        CHECK(errLocal.str() == "");
        REQUIRE(status == StatusCode::SUCCESS);
        // distributed run
        auto envVar = "DISTRIBUTED_WORKERS";
        std::stringstream outDist;
        std::stringstream errDist;
        setenv(envVar, distWorkerStr.c_str(), 1);
        status = runDaphne(outDist, errDist, filename.c_str());
        unsetenv(envVar);
        CHECK(errDist.str() == "");
        REQUIRE(status == StatusCode::SUCCESS);

        CHECK(outLocal.str() == outDist.str());
    }
}