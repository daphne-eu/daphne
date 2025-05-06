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

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "AsyncUtil.h"
#include "IO_URing.h"

void CompletionWrapper(std::atomic<bool> *shut_down_requested, URing *ring, std::atomic<bool> *sleep_cv,
                       uint64_t *min_idle_time_till_sleep_in_ms);
void SubmissionWrapper(std::atomic<bool> *shut_down_requested, URing *ring, std::atomic<bool> *sleep_cv,
                       std::atomic<bool> *sleep_cv_of_completion_thread, uint64_t *min_idle_time_till_sleep_in_ms);

struct URingRunner {
    URing ring;
    uint64_t min_idle_time;
    std::atomic<bool> shut_down_requested;
    std::thread submission_worker;
    std::thread completion_worker;
    std::atomic<bool> submission_worker_should_be_active;
    std::atomic<bool> completion_worker_should_be_active;
    URingRunner(uint32_t ring_size, bool use_io_dev_polling, bool use_sq_polling,
                uint32_t io_uring_submission_queue_idle_timeout_in_ms, uint64_t idle_time_till_sleep_in_ms = 20);
    ~URingRunner();
};

// Manages amount_of_io_urings instances of io_uring, that each come with 2 dedicated threads for submission and
// completion handling. Users may simply call submitX() and then can check the status of their i/o requests via the
// respective IO_STATUS returned from submitX(). This interface removes need for the user to address the thread safety
// concerns that come with accessing an io_uring instance from multiple threads, out-of-order arrival of requests /
// observing a cqe on one thread that belongs to a i/o request originating from another thread, distributing requests
// among instances, keeping track to which instance each request was submitted and store their meta data while the
// requests remain in flight etc...
// Note: Currently the threads are spawned and "owned" by the threadpool and go to sleep if there is/recently has not
// been more work for them to do. For daphne it would probably be better if the threads would not be managed internally
// and instead be provided by daphne, simply by calling Completion/SubmissionWrapper() (which should be changed to
// simply return rather than go to sleep in times of low load).
struct IOThreadpool {
    std::vector<URingRunner *> runners;
    IOThreadpool(uint32_t amount_of_io_urings, uint32_t ring_size, bool use_io_dev_polling, bool use_sq_polling,
                 uint32_t io_uring_submission_queue_idle_timeout_in_ms, uint64_t idle_time_till_sleep_in_ms = 20);
    ~IOThreadpool();
    std::unique_ptr<std::atomic<IO_STATUS>[]> SubmitReads(const std::vector<URingRead> &reads);
    std::unique_ptr<std::atomic<IO_STATUS>[]> SubmitWrites(const std::vector<URingWrite> &writes);
    void SubmitReads(const std::vector<URingRead> &reads, std::atomic<IO_STATUS> *results);
    void SubmitWrites(const std::vector<URingWrite> &writes, std::atomic<IO_STATUS> *results);
};
