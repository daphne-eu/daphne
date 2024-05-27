/*
 * Copyright 2022 The DAPHNE Consortium
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

#include <cstddef>
#include <cstdint>
#include <vector>
#include <chrono>

#include <fcntl.h>

#include <tags.h>

#include <catch.hpp>

#include "runtime/local/io/io_uring/AsyncUtil.h"
#include "runtime/local/io/io_uring/IO_Threadpool.h"
#include "runtime/local/io/io_uring/IO_URing.h"

TEST_CASE("io_uring / io threadpool setup Tests", TAG_IO) {
    URing(64, false, false, 1000);

    { IOThreadpool(1, 64, false, false, 1000); }
    { IOThreadpool(4, 64, false, false, 1000); }
    { IOThreadpool(2, 64, false, false, 1000); }
    { IOThreadpool(8, 64, false, false, 1000); }
}

TEST_CASE("io_uring basic File R/W test", TAG_IO) {
    IOThreadpool io_pool(1, 64, false, false, 1000);

    int fd = open("./test/runtime/local/io/uring_test_file", O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, S_IRUSR | S_IWUSR);
    REQUIRE(fd > 0);

    auto data_page   = std::make_unique<uint64_t[]>(4096 / sizeof(uint64_t));
    auto result_page = std::make_unique<uint64_t[]>(4096 / sizeof(uint64_t));
    for (size_t i = 0; i < (4096 / sizeof(uint64_t)); i++) {
        data_page[i]   = 42;
        result_page[i] = 0;
    }

    // The write
    std::vector<URingWrite> write_requests;
    write_requests.push_back({data_page.get(), 4096, 0, fd});

    std::unique_ptr<std::atomic<IO_STATUS>[]> write_future = io_pool.SubmitWrites(write_requests);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    bool timed_out = false;
    while (write_future[0] == IO_STATUS::IN_FLIGHT) {
        if (static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                .count()) > 3000) {
            timed_out = true;
            break;
        }
    }
    REQUIRE(!timed_out);

    REQUIRE(write_future[0] == IO_STATUS::SUCCESS);

    // The Read
    std::vector<URingRead> read_requests;
    read_requests.push_back({result_page.get(), 4096, 0, fd});

    std::unique_ptr<std::atomic<IO_STATUS>[]> read_future = io_pool.SubmitReads(read_requests);

    timed_out = false;
    start     = std::chrono::high_resolution_clock::now();
    while (read_future[0] == IO_STATUS::IN_FLIGHT) {
        if (static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                .count()) > 3000) {
            timed_out = true;
            break;
        }
    }
    REQUIRE(!timed_out);

    REQUIRE(read_future[0] == IO_STATUS::SUCCESS);

    for (size_t i = 0; i < (4096 / sizeof(uint64_t)); i++) {
        REQUIRE(result_page[i] == 42);
    }
}

TEST_CASE("io_uring sleep cv test", TAG_IO) {
    IOThreadpool io_pool(1, 64, false, false, 1000);

    int fd = open("./test/runtime/local/io/uring_test_file", O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, S_IRUSR | S_IWUSR);
    REQUIRE(fd > 0);

    auto data_page   = std::make_unique<uint64_t[]>(4096 / sizeof(uint64_t));
    auto result_page = std::make_unique<uint64_t[]>(4096 / sizeof(uint64_t));
    for (size_t i = 0; i < (4096 / sizeof(uint64_t)); i++) {
        data_page[i]   = 42;
        result_page[i] = 0;
    }
    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Wait till threads timed out and went to sleep. Should happen after roughly idle_:timeout_threshold = 20ms 
    while(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                .count()) < 500) {};
    
    // Threads should be sleeping now an actual assert does not make sense though due to spurious wakeups

    // The write
    std::vector<URingWrite> write_requests;
    write_requests.push_back({data_page.get(), 4096, 0, fd});

    // Now do read and threads should wake up and everything should just work normally
    std::unique_ptr<std::atomic<IO_STATUS>[]> write_future = io_pool.SubmitWrites(write_requests);
    start = std::chrono::high_resolution_clock::now();

    bool timed_out = false;
    while (write_future[0] == IO_STATUS::IN_FLIGHT) {
        if (static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                .count()) > 3000) {
            timed_out = true;
            break;
        }
    }
    REQUIRE(!timed_out);

    REQUIRE(write_future[0] == IO_STATUS::SUCCESS);

    // The Read
    std::vector<URingRead> read_requests;
    read_requests.push_back({result_page.get(), 4096, 0, fd});

    start = std::chrono::high_resolution_clock::now();

    // Let the threads go to sleep again
    while(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                .count()) < 500) {};

    std::unique_ptr<std::atomic<IO_STATUS>[]> read_future = io_pool.SubmitReads(read_requests);

    timed_out = false;
    start     = std::chrono::high_resolution_clock::now();
    while (read_future[0] == IO_STATUS::IN_FLIGHT) {
        if (static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                .count()) > 3000) {
            timed_out = true;
            break;
        }
    }
    REQUIRE(!timed_out);

    REQUIRE(read_future[0] == IO_STATUS::SUCCESS);

    for (size_t i = 0; i < (4096 / sizeof(uint64_t)); i++) {
        REQUIRE(result_page[i] == 42);
    }
}

TEST_CASE("io_uring basic File R/W test - multiple instance and requests", TAG_IO) {
    IOThreadpool io_pool(7, 64, false, false, 1000);

    int fd = open("./test/runtime/local/io/uring_test_file", O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, S_IRUSR | S_IWUSR);
    REQUIRE(fd > 0);

    uint64_t op_count = 47;

    auto data_page   = std::make_unique<uint8_t[]>(op_count * 4096);
    auto result_page = std::make_unique<uint8_t[]>(op_count * 4096);

    for (size_t i = 0; i < 4096 * op_count; i++) {
        data_page[i]   = (i * 42) % 256;
        result_page[i] = 0;
    }

    // The write
    std::vector<URingWrite> write_requests;
    for (size_t i=0; i < op_count; i++) {
        write_requests.push_back({&data_page[i * 4096], 4096, i * 4096, fd});
    }
    std::unique_ptr<std::atomic<IO_STATUS>[]> write_future = io_pool.SubmitWrites(write_requests);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    bool timed_out = false;
    for (size_t i=0; i < op_count; i++) {
        while (write_future[i] == IO_STATUS::IN_FLIGHT) {
            if (static_cast<uint64_t>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                    .count()) > 3000) {
                timed_out = true;
                break;
            }
        }
    }
    REQUIRE(!timed_out);

    for (size_t i=0; i < op_count; i++) {
        REQUIRE(write_future[i] == IO_STATUS::SUCCESS);
    }

    // The Read
    std::vector<URingRead> read_requests;
    for (size_t i=0; i < op_count; i++) {
        read_requests.push_back({&result_page[i * 4096], 4096, i * 4096, fd});
    }

    std::unique_ptr<std::atomic<IO_STATUS>[]> read_future = io_pool.SubmitReads(read_requests);

    timed_out = false;
    start     = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i < op_count; i++) {
        while (read_future[i] == IO_STATUS::IN_FLIGHT) {
            if (static_cast<uint64_t>(
                  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
                    .count()) > 3000) {
                timed_out = true;
                break;
            }
        }
    }
    REQUIRE(!timed_out);

    for (size_t i=0; i < op_count; i++) {
        REQUIRE(read_future[i] == IO_STATUS::SUCCESS);
    }

    for (size_t i = 0; i < op_count * 4096; i++) {
        REQUIRE(result_page[i] == (i * 42) % 256);
    }
}
