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
#include <fcntl.h>

#include <tags.h>

#include <catch.hpp>

#include <grpcpp/grpcpp.h>

#include<thread>

const std::string dirPath = "test/api/cli/distributed/";

TEST_CASE("Distributed runtime tests using gRPC", TAG_DISTRIBUTED)
{
    auto addr1 = "0.0.0.0:50051";
    auto addr2 = "0.0.0.0:50052";    
    // Redirect worker output to null
    int nullFd = open("/dev/null", O_WRONLY);
    auto pid1 = runProgramInBackground(nullFd, nullFd, "bin/DistributedWorker", "DistributedWorker", addr1);
    auto pid2 = runProgramInBackground(nullFd, nullFd, "bin/DistributedWorker", "DistributedWorker", addr2);
    assert(std::getenv("DISTRIBUTED_WORKERS") == nullptr);
    auto distWorkerStr = std::string(addr1) + ',' + addr2;

    SECTION("Execution of scripts using distributed runtime (gRPC)"){
        // TODO Make these script individual DYNAMIC_SECTIONs.
        for (auto i = 1u; i <= 4; ++i) {
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
            status = runDaphne(outDist, errDist, std::string("--distributed").c_str(), std::string("--dist_backend=sync-gRPC").c_str(), filename.c_str());
            unsetenv(envVar);
            CHECK(errDist.str() == "");
            REQUIRE(status == StatusCode::SUCCESS);

            CHECK(outLocal.str() == outDist.str());
        }
    }
    SECTION("Distributed chunked messages (gRPC)"){
        
        auto filename = dirPath + "distributed_2.daphne";

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
        status = runDaphne(outDist, errDist, std::string("--max-distr-chunk-size=100").c_str(), std::string("--distributed").c_str(), std::string("--dist_backend=sync-gRPC").c_str(), filename.c_str());
        unsetenv(envVar);
        CHECK(errDist.str() == "");
        REQUIRE(status == StatusCode::SUCCESS);

        CHECK(outLocal.str() == outDist.str());
    
    }
    // SECTION("Distributed read operation"){
    //     auto filenameLocal = dirPath + "distributedRead/readLocalMat.daphne";
    //     auto filenameDistr = dirPath + "distributedRead/readDistrMat.daphne";

    //     std::stringstream outLocal;
    //     std::stringstream errLocal;
    //     int status = runDaphne(outLocal, errLocal, filenameLocal.c_str());

    //     CHECK(errLocal.str() == "");
    //     REQUIRE(status == StatusCode::SUCCESS);
    //     // distributed run
    //     auto envVar = "DISTRIBUTED_WORKERS";
    //     std::stringstream outDist;
    //     std::stringstream errDist;
    //     setenv(envVar, distWorkerStr.c_str(), 1);
    //     status = runDaphne(outDist, errDist, std::string("--vec").c_str(), filenameDistr.c_str());
    //     unsetenv(envVar);
    //     CHECK(errDist.str() == "");
    //     REQUIRE(status == StatusCode::SUCCESS);

    //     CHECK(outLocal.str() == outDist.str());
    // }

    kill(pid1, SIGKILL);
    kill(pid2, SIGKILL);
    wait(NULL);   
}

#ifdef USE_MPI
TEST_CASE("Distributed runtime tests using MPI", TAG_DISTRIBUTED)
{

    SECTION("Execution of scripts using distributed runtime (MPI)"){
        // TODO Make these script individual DYNAMIC_SECTIONs.

        for (auto i = 1u; i <= 4; ++i) {
            auto filename = dirPath + "distributed_" + std::to_string(i) + ".daphne";
           
            std::stringstream outLocal;
            std::stringstream errLocal;
            int status = runDaphne(outLocal, errLocal, filename.c_str());
           
            CHECK(errLocal.str() == "");
            REQUIRE(status == StatusCode::SUCCESS);

            std::stringstream outDist;
            std::stringstream errDist;
            status = runProgram(outDist, errDist, "mpirun", "--allow-run-as-root", "-np", "4", "bin/daphne", "--distributed", "--dist_backend=MPI", filename.c_str());
           
            CHECK(errDist.str() == "");
            REQUIRE(status == StatusCode::SUCCESS);

            CHECK(outLocal.str() == outDist.str());
        }
    }
    SECTION("Distributed chunked messages (MPI)"){

        auto filename = dirPath + "distributed_2.daphne";

        std::stringstream outLocal;
        std::stringstream errLocal;
       
        int status = runDaphne(outLocal, errLocal, filename.c_str());
        CHECK(errLocal.str() == "");
        REQUIRE(status == StatusCode::SUCCESS);
        
        std::stringstream outDist;
        std::stringstream errDist;
        status = runProgram(outDist, errDist,  "mpirun", "--allow-run-as-root", "-np", "4", "bin/daphne", "--distributed", "--dist_backend=MPI", "--max-distr-chunk-size=100", filename.c_str());
        CHECK(errDist.str() == "");
        REQUIRE(status == StatusCode::SUCCESS);

        CHECK(outLocal.str() == outDist.str());
    
    }
    wait(NULL);   
}
#endif