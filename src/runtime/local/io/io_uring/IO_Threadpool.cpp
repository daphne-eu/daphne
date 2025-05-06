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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>

#include "AsyncUtil.h"
#include "IO_Threadpool.h"
#include "IO_URing.h"

void CompletionWrapper(std::atomic<bool> *shut_down_requested, URing *ring, std::atomic<bool> *sleep_cv,
                       uint64_t *min_idle_time_till_sleep_in_ms) {
    std::chrono::time_point<std::chrono::high_resolution_clock> initially_idle_timestamp;
    bool idle = false;

    while (!(*shut_down_requested)) {
        bool proccessed_a_cqe = false;
        // The PeekCQAndHandleCQEs() call contains the actual work to be performed. As both the checks whether to shut
        // down and or to sleep require "relatively" expensive seq_cst atomics they are only check once in a while i.e.
        // here every min_cqe_processing amount of times
        constexpr size_t min_cqe_processing = 100;
        for (size_t i = 0; i < min_cqe_processing; i++) {
            if (ring->PeekCQAndHandleCQEs()) {
                proccessed_a_cqe = true;
            }
        }

        // we do this check here over directly going to check the q sizes since that involves atomic ops
        if (!proccessed_a_cqe) {
            // Check if any request are in flight
            bool request_in_flight = (static_cast<int32_t>(ring->total_slots) == ring->remaining_slots.load());

            if (request_in_flight) {
                if (!idle) { // No requests in flight -> begin timeout if not already
                    idle = true;
                    initially_idle_timestamp = std::chrono::high_resolution_clock::now();
                } else { // Check if timed out y ? -> go to sleep
                    if (static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  std::chrono::high_resolution_clock::now() - initially_idle_timestamp)
                                                  .count()) >= *min_idle_time_till_sleep_in_ms) {
                        *sleep_cv = false;
                        sleep_cv->wait(false, std::memory_order_seq_cst);
                        idle = false;
                    }
                }
            } else { // we processed a cqe -> stop timer
                idle = false;
            }
        }
    }
}

void SubmissionWrapper(std::atomic<bool> *shut_down_requested, URing *ring, std::atomic<bool> *sleep_cv,
                       std::atomic<bool> *sleep_cv_of_completion_thread, uint64_t *min_idle_time_till_sleep_in_ms) {
    std::chrono::time_point<std::chrono::high_resolution_clock> initially_idle_timestamp;
    bool idle = false;

    while (!(*shut_down_requested)) {
        // More precisely it also tracks the attempt to submit requests i.e. it indicates whether we submitted smth or
        // tried but failed. If it is true either there are still the failed requests in the queue to submit again or we
        // submitted
        // some requests last time -> likely that there are more and if not this should fail next round. -> this is an
        // indicator
        // if there is more work to be done or not without directly checking atomic q sizes
        bool submitted_a_request = false;
        // The PeekCQAndHandleCQEs() call contains the actual work to be performed. As both the checks whether to shut
        // down and or to sleep require "relatively" expensive seq_cst atomics they are only check once in a while i.e.
        // here every min_cqe_processing amount of times
        constexpr size_t min_cqe_processing = 100;
        for (size_t i = 0; i < min_cqe_processing; i++) {
            if (ring->SubmitRead() || ring->SubmitWrite()) {
                submitted_a_request = true;
            }
        }

        if (submitted_a_request) {
            // wakeup completion thread if if sleeps and we submitted requests
            bool expected_value_if_completion_thread_sleeps = false;
            if (sleep_cv_of_completion_thread->compare_exchange_strong(expected_value_if_completion_thread_sleeps, true,
                                                                       std::memory_order_seq_cst)) {
                sleep_cv_of_completion_thread->notify_one();
            }
        }

        if (!submitted_a_request) {
            if (!idle) { // did not submit/try to submit a request -> begin timeout if not already
                idle = true;
                initially_idle_timestamp = std::chrono::high_resolution_clock::now();
            } else { // Already started timer -> check if timed out -> go to sleep
                if (static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::high_resolution_clock::now() - initially_idle_timestamp)
                                              .count()) >= *min_idle_time_till_sleep_in_ms) {
                    *sleep_cv = false;
                    sleep_cv->wait(false, std::memory_order_seq_cst);
                    idle = false;
                }
            }
        } else { // we submitted a request -> stop timer
            idle = false;
        }
    }
}

URingRunner::URingRunner(
    uint32_t ring_size, bool use_io_dev_polling, bool use_sq_polling,
    uint32_t io_uring_submission_queue_idle_timeout_in_ms, // concerns the kernel space sq poll thread
    uint64_t idle_time_till_sleep_in_ms)                   // concerns the user space threads
    : ring(ring_size, use_io_dev_polling, use_sq_polling, io_uring_submission_queue_idle_timeout_in_ms),
      min_idle_time(idle_time_till_sleep_in_ms), shut_down_requested(false),
      submission_worker(SubmissionWrapper, &shut_down_requested, &ring, &submission_worker_should_be_active,
                        &completion_worker_should_be_active, &min_idle_time),
      completion_worker(CompletionWrapper, &shut_down_requested, &ring, &completion_worker_should_be_active,
                        &min_idle_time),
      submission_worker_should_be_active(true), completion_worker_should_be_active(true){};

URingRunner::~URingRunner() {
    shut_down_requested = true;
    submission_worker_should_be_active = true;
    submission_worker_should_be_active.notify_all();
    submission_worker.join();
    completion_worker_should_be_active = true;
    completion_worker_should_be_active.notify_all();
    completion_worker.join();
}

IOThreadpool::IOThreadpool(
    uint32_t amount_of_io_urings, uint32_t ring_size, bool use_io_dev_polling, bool use_sq_polling,
    uint32_t submission_queue_idle_timeout_in_ms, // concerns the kernel space thread sq poll thread
    uint64_t idle_time_till_sleep_in_ms) {        // concerns the user space threads
    runners.resize(amount_of_io_urings);
    for (uint32_t i = 0; i < amount_of_io_urings; i++) {
        runners[i] =
            new URingRunner(ring_size, use_io_dev_polling, use_sq_polling, submission_queue_idle_timeout_in_ms);
    }
}
IOThreadpool::~IOThreadpool() {
    for (size_t i = 0; i < runners.size(); i++) {
        delete (runners[i]);
    }
}

std::unique_ptr<std::atomic<IO_STATUS>[]> IOThreadpool::SubmitReads(const std::vector<URingRead> &reads) {
    std::unique_ptr<std::atomic<IO_STATUS>[]> results = std::make_unique<std::atomic<IO_STATUS>[]>(reads.size());

    IOThreadpool::SubmitReads(reads, results.get());

    return results;
}

std::unique_ptr<std::atomic<IO_STATUS>[]> IOThreadpool::SubmitWrites(const std::vector<URingWrite> &writes) {
    std::unique_ptr<std::atomic<IO_STATUS>[]> results = std::make_unique<std::atomic<IO_STATUS>[]>(writes.size());

    IOThreadpool::SubmitWrites(writes, results.get());

    return results;
}

void IOThreadpool::SubmitReads(const std::vector<URingRead> &reads, std::atomic<IO_STATUS> *results) {
    uint64_t reads_per_ring =
        reads.size() % runners.size() == 0 ? reads.size() / runners.size() : 1 + (reads.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
        uint64_t current_read_batch =
            (reads.size() - current_offset) > reads_per_ring ? reads_per_ring : (reads.size() - current_offset);

        std::vector<URingReadInternal> ring_read_batch;
        ring_read_batch.resize(current_read_batch);
        for (uint64_t j = 0; j < current_read_batch; j++) {
            ring_read_batch[j] = {reads[j + current_offset].dest, reads[j + current_offset].dest,
                                  reads[j + current_offset].size, reads[j + current_offset].offset,
                                  reads[j + current_offset].fd,   &results[j + current_offset]};
            results[j + current_offset] = IO_STATUS::IN_FLIGHT;
        }

        current_offset += current_read_batch;

        runners[i]->ring.Enqueue(ring_read_batch);

        // wakeup submission thread if required
        bool expected_value_if_submission_thread_sleeps = false;
        if (runners[i]->submission_worker_should_be_active.compare_exchange_strong(
                expected_value_if_submission_thread_sleeps, true, std::memory_order_seq_cst)) {
            runners[i]->submission_worker_should_be_active.notify_one();
        }

        if (current_offset >= reads.size()) {
            break;
        }
    }
}

void IOThreadpool::SubmitWrites(const std::vector<URingWrite> &writes, std::atomic<IO_STATUS> *results) {
    uint64_t writes_per_ring =
        writes.size() % runners.size() == 0 ? writes.size() / runners.size() : 1 + (writes.size() / runners.size());

    uint64_t current_offset = 0;
    for (size_t i = 0; i < runners.size(); i++) {
        uint64_t current_write_batch =
            (writes.size() - current_offset) > writes_per_ring ? (writes.size() - current_offset) : writes_per_ring;
        std::vector<URingWriteInternal> ring_write_batch;
        ring_write_batch.resize(current_write_batch);

        for (uint64_t j = 0; j < current_write_batch; j++) {
            ring_write_batch[j] = {writes[j + current_offset].src,  writes[j + current_offset].src,
                                   writes[j + current_offset].size, writes[j + current_offset].offset,
                                   writes[j + current_offset].fd,   &results[j + current_offset]};
            results[j + current_offset] = IO_STATUS::IN_FLIGHT;
        }

        current_offset += current_write_batch;

        runners[i]->ring.Enqueue(ring_write_batch);

        // wakeup submission thread if required
        bool expected_value_if_submission_thread_sleeps = false;
        if (runners[i]->submission_worker_should_be_active.compare_exchange_strong(
                expected_value_if_submission_thread_sleeps, true, std::memory_order_seq_cst)) {
            runners[i]->submission_worker_should_be_active.notify_one();
        }

        if (current_offset >= writes.size()) {
            break;
        }
    }
}
