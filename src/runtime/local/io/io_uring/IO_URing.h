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

#include <asm-generic/errno-base.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include <liburing.h>

#include "Container.h"
#include "AsyncUtil.h"

enum struct IO_OP_CODE : uint8_t { READ = 0, WRITE = 1 };

struct URingRead {
    void *dest;
    uint64_t size;
    uint64_t offset;
    int fd;
};

struct URingReadInternal {
    void *initial_dest;
    void *current_dest;
    uint64_t remaining_size;
    uint64_t offset;
    int fd;
    std::atomic<IO_STATUS> *status;
};

struct URingWrite {
    void *src;
    uint64_t size;
    uint64_t offset;
    int fd;
};

struct URingWriteInternal {
    void *initial_dest;
    void *current_dest;
    uint64_t remaining_size;
    uint64_t offset;
    int fd;
    std::atomic<IO_STATUS> *status;
};

struct InFilghtSQE {
    void *initial;
    void *current;
    uint64_t remaining_size;
    void *result;
    uint64_t offset;
    int fd;
    IO_OP_CODE op_code;
};

// Simple wrapper class around io_uring supporting basic operations like read()
// or write(), without support for "advanced" features like
// registered/lightweight fds, registered buffers, multi-shot variants of
// io-related syscalls, canceling of SQEs and draining of the rings, network
// related io, general multi threading support
// Thread safety: io_urings default thread safety model intends at a maximum one
// user space thread to operate on the SQ and CQ respectively.
struct URing {
    bool use_io_dev_polling;    // Needs hardware support
    bool use_sq_polling;

    struct io_uring_params ring_para;
    struct io_uring ring;
    int32_t ring_fd;

    uint32_t total_slots;
    std::atomic<int32_t> remaining_slots;

    ThreadSafeStack<URingReadInternal> read_submission_q;
    ThreadSafeStack<URingWriteInternal> write_submission_q;

    Pool<InFilghtSQE> in_flight_SQEs;

    // ring_size must be <= 32K and will be rounded up to the next power of two
    URing(uint32_t ring_size,
          bool use_io_dev_polling,
          bool use_sq_polling,
          uint32_t submission_queue_idle_timeout_in_ms);

    ~URing();

    void Enqueue(const std::vector<URingReadInternal> &reads);
    void Enqueue(const std::vector<URingWriteInternal> &writes);
    bool SubmitRead();
    bool SubmitWrite();
    bool PeekCQAndHandleCQEs();

    uint64_t GetUUID(IO_OP_CODE op_code, uint64_t slot_id);
    uint64_t GetSlotIdFromUUID(uint64_t uuid);
    IO_OP_CODE GetOPCodeFromUUID(uint64_t uuid);
    void HandleRead(uint64_t slot_id, int32_t cqe_res);
    void HandleWrite(uint64_t slot_id, int32_t cqe_res);
};
