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
#include <cstdint>
#include <cstring>
#include <vector>

#include <asm-generic/errno-base.h>
#include <liburing.h>

#include "AsyncUtil.h"
#include "IO_URing.h"

// ring_size must be <= 32K and will be rounded up to the next power of two
URing::URing(uint32_t ring_size, bool use_io_dev_polling, bool use_sq_polling,
             uint32_t submission_queue_idle_timeout_in_ms)
    : use_io_dev_polling(use_io_dev_polling), use_sq_polling(use_sq_polling), total_slots(ring_size),
      remaining_slots(ring_size), in_flight_SQEs(ring_size) {
    std::memset(&ring_para, 0, sizeof(ring_para));

    if (use_io_dev_polling) {
        ring_para.flags |= IORING_SETUP_IOPOLL;
    }
    if (use_sq_polling) {
        ring_para.flags |= IORING_SETUP_SQPOLL;
    }

    ring_para.sq_thread_idle = submission_queue_idle_timeout_in_ms;

    int status = io_uring_queue_init_params(ring_size, &ring, &ring_para);
    if (status != 0) {
        std::abort();
    }

    ring_fd = ring.ring_fd;
}

URing::~URing() { io_uring_queue_exit(&ring); }

void URing::Enqueue(const std::vector<URingReadInternal> &reads) { read_submission_q.Push(reads); }

void URing::Enqueue(const std::vector<URingWriteInternal> &writes) { write_submission_q.Push(writes); }

uint64_t URing::GetUUID(IO_OP_CODE op_code, uint64_t slot_id) { return (slot_id << 8) + static_cast<uint8_t>(op_code); }

uint64_t URing::GetSlotIdFromUUID(uint64_t uuid) { return uuid >> 8; }

IO_OP_CODE URing::GetOPCodeFromUUID(uint64_t uuid) {
    return static_cast<IO_OP_CODE>(static_cast<uint8_t>(uuid & 0xff));
}

bool URing::SubmitRead() {
    constexpr uint64_t read_batch_size = 64;

    std::vector<URingReadInternal> reads = read_submission_q.TryPop(read_batch_size);
    uint64_t amount_of_requests_to_submit = reads.size();

    int32_t slots_before_sub = remaining_slots.fetch_sub(amount_of_requests_to_submit);
    if (slots_before_sub < static_cast<int64_t>(amount_of_requests_to_submit)) {
        if (slots_before_sub < 0) {
            remaining_slots += amount_of_requests_to_submit;
        } else {
            int32_t amount_to_put_back = amount_of_requests_to_submit - slots_before_sub;
            remaining_slots += amount_to_put_back;
            amount_of_requests_to_submit -= amount_to_put_back;
        }
    }

    uint64_t requests_submitted = 0;
    std::vector<uint64_t> alloced_slots_for_sqe_meta_data;
    alloced_slots_for_sqe_meta_data.reserve(amount_of_requests_to_submit);

    // Attempt to get a sqe from io_uring for all requests we wish to submit.
    // Which may not be possible.
    for (size_t i = 0; i < amount_of_requests_to_submit; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (sqe == nullptr) { // no sqe available
            break;
        }

        alloced_slots_for_sqe_meta_data.push_back(
            in_flight_SQEs.Insert({reads[i].initial_dest, reads[i].current_dest, reads[i].remaining_size,
                                   reads[i].status, reads[i].offset, reads[i].fd, IO_OP_CODE::READ}));

        io_uring_prep_read(sqe, reads[i].fd, reads[i].current_dest, reads[i].remaining_size, reads[i].offset);
        io_uring_sqe_set_data64(sqe, static_cast<__u64>(GetUUID(IO_OP_CODE::READ, alloced_slots_for_sqe_meta_data[i])));

        // Workaround for https://github.com/axboe/liburing/issues/88
        // In the case io_uriong_submit() partially or fully succeeds, we can not
        // trust the returned positive value to be accurate when using SQ polling.
        // It is impossible to distinguish partial from complete success -> always
        // just submit 1 -> we know which one failed
        if (use_sq_polling) {
            int amount_sqe_submitted = io_uring_submit(&ring);
            if (amount_sqe_submitted < 0) {
                in_flight_SQEs.template Free<false>(alloced_slots_for_sqe_meta_data[i]);
                break;
            }
            requests_submitted++;
        }
    }

    if (!use_sq_polling) {
        // Now for all requests for which we got a sqe tell io_uring about them
        uint64_t sqes_to_submit = alloced_slots_for_sqe_meta_data.size();
        uint32_t fruitless_attempts = 0;
        // io_uring_submit() is allowed to not process all requests at once (including processing 0).
        // Under normal operation and if no error occurred, io_uring will process multiple requests (frequently all)
        // with one call.
        // This constant provides a bound for maximum amount of attempts that returned 0 that have occurred in a row, as
        // a fail safe against a hang in this loop.
        constexpr uint32_t max_fruitless_attempts = 5;

        while (requests_submitted != sqes_to_submit) {
            int amount_sqe_submitted = io_uring_submit(&ring);

            if (amount_sqe_submitted > 0) {
                if ((static_cast<uint32_t>(amount_sqe_submitted) + requests_submitted) > sqes_to_submit) {
                    std::abort();
                }
                requests_submitted += static_cast<uint32_t>(amount_sqe_submitted);
                fruitless_attempts = 0;
            } else {
                if (amount_sqe_submitted < 0) {
                    break;
                }
                if (amount_sqe_submitted == 0) {
                    fruitless_attempts++;
                    if (fruitless_attempts > max_fruitless_attempts) {
                        break;
                    }
                    continue;
                }
            }
        }
    }

    if (requests_submitted < amount_of_requests_to_submit) {
        for (size_t i = requests_submitted; i < amount_of_requests_to_submit; i++) {
            in_flight_SQEs.template Free<false>(alloced_slots_for_sqe_meta_data[i]);
        }
    }

    // Inform if there was !an attempt! to make progress -> use amount_of_requests_to_submit over requests_submitted
    return amount_of_requests_to_submit != 0;
}

bool URing::SubmitWrite() {
    constexpr uint64_t write_batch_size = 64;

    std::vector<URingWriteInternal> writes = write_submission_q.TryPop(write_batch_size);
    uint64_t amount_of_requests_to_submit = writes.size();

    int32_t slots_before_sub = remaining_slots.fetch_sub(amount_of_requests_to_submit);
    if (slots_before_sub < static_cast<int64_t>(amount_of_requests_to_submit)) {
        if (slots_before_sub < 0) {
            remaining_slots += amount_of_requests_to_submit;
        } else {
            int32_t amount_to_put_back = amount_of_requests_to_submit - slots_before_sub;
            remaining_slots += amount_to_put_back;
            amount_of_requests_to_submit -= amount_to_put_back;
        }
    }

    uint64_t requests_submitted = 0;
    std::vector<uint64_t> alloced_slots_for_sqe_meta_data;
    alloced_slots_for_sqe_meta_data.reserve(amount_of_requests_to_submit);

    // Attempt to get a sqe from io_uring for all requests we wish to submit.
    // Which may not be possible.
    for (size_t i = 0; i < amount_of_requests_to_submit; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (sqe == nullptr) { // no sqe available
            break;
        }
        alloced_slots_for_sqe_meta_data.push_back(
            in_flight_SQEs.Insert({writes[i].initial_dest, writes[i].current_dest, writes[i].remaining_size,
                                   writes[i].status, writes[i].offset, writes[i].fd, IO_OP_CODE::WRITE}));

        io_uring_prep_write(sqe, writes[i].fd, writes[i].initial_dest, writes[i].remaining_size, writes[i].offset);
        io_uring_sqe_set_data64(sqe,
                                static_cast<__u64>(GetUUID(IO_OP_CODE::WRITE, alloced_slots_for_sqe_meta_data[i])));

        // Workaround for https://github.com/axboe/liburing/issues/88
        // In the case io_uriong_submit() partially or fully succeeds, we can not
        // trust the returned positive value to be accurate when using SQ polling.
        // It is impossible to distinguish partial from complete success -> always
        // just submit 1 -> we know which one failed
        if (use_sq_polling) {
            int amount_sqe_submitted = io_uring_submit(&ring);
            if (amount_sqe_submitted < 0) {
                in_flight_SQEs.template Free<false>(alloced_slots_for_sqe_meta_data[i]);
                break;
            }
            requests_submitted++;
        }
    }

    if (!use_sq_polling) {
        // Now for all requests for which we got a sqe tell io_uring about them
        uint64_t sqes_to_submit = alloced_slots_for_sqe_meta_data.size();
        uint32_t fruitless_attempts = 0;
        // io_uring_submit() is allowed to not process all requests at once (including processing 0).
        // Under normal operation and if no error occurred, io_uring will process multiple requests (frequently all)
        // with one call.
        // This constant provides a bound for maximum amount of attempts that returned 0 that have occurred in a row, as
        // a fail safe against a hang in this loop.
        constexpr uint32_t max_fruitless_attempts = 5;

        while (requests_submitted != sqes_to_submit) {
            int amount_sqe_submitted = io_uring_submit(&ring);

            if (amount_sqe_submitted > 0) {
                if ((static_cast<uint32_t>(amount_sqe_submitted) + requests_submitted) > sqes_to_submit) {
                    std::abort();
                }
                requests_submitted += static_cast<uint32_t>(amount_sqe_submitted);
                fruitless_attempts = 0;
            } else {
                if (amount_sqe_submitted < 0) {
                    break;
                }
                if (amount_sqe_submitted == 0) {
                    fruitless_attempts++;
                    if (fruitless_attempts > max_fruitless_attempts) {
                        break;
                    }
                    continue;
                }
            }
        }
    }

    if (requests_submitted < amount_of_requests_to_submit) {
        for (size_t i = requests_submitted; i < amount_of_requests_to_submit; i++) {
            in_flight_SQEs.template Free<false>(alloced_slots_for_sqe_meta_data[i]);
        }
    }

    // Inform if there was !an attempt! to make progress -> use amount_of_requests_to_submit over requests_submitted
    return amount_of_requests_to_submit != 0;
}

void URing::HandleRead(uint64_t slot_id, int32_t cqe_res) {
    InFilghtSQE in_flight_request = this->in_flight_SQEs.Extract(slot_id);

    std::atomic<IO_STATUS> *status = static_cast<std::atomic<IO_STATUS> *>(in_flight_request.result);

    // Either request fully or partially fulfilled
    if (cqe_res > 0) {
        // partial read -> not a failure -> resubmit modified request
        if (static_cast<uint32_t>(cqe_res) < in_flight_request.remaining_size) {
            this->read_submission_q.Push(
                {in_flight_request.initial, (static_cast<uint8_t *>(in_flight_request.current) + cqe_res),
                 in_flight_request.remaining_size - cqe_res, in_flight_request.offset, in_flight_request.fd,
                 reinterpret_cast<std::atomic<IO_STATUS> *>(in_flight_request.result)});
            return;
        }

        *status = IO_STATUS::SUCCESS;
        return;
    }

    // Zero progress returns are also considered errors
    switch (-cqe_res) {
    case EIO:
        *status = IO_STATUS::IO_ERROR;
        return;
    case EFAULT:
        *status = IO_STATUS::ACCESS_DENIED;
        return;
    case EBADF:
        *status = IO_STATUS::BAD_FD;
        return;
    default:
        *status = IO_STATUS::OTHER_ERROR;
        return;
    }
}

void URing::HandleWrite(uint64_t slot_id, int32_t cqe_res) {
    InFilghtSQE in_flight_request = this->in_flight_SQEs.Extract(slot_id);
    std::atomic<IO_STATUS> *result = static_cast<std::atomic<IO_STATUS> *>(in_flight_request.result);

    // Either request fully or partially fulfilled
    if (cqe_res > 0) {
        // partial write -> not a failure -> resubmit modified request
        if (static_cast<uint32_t>(cqe_res) < in_flight_request.remaining_size) {
            this->write_submission_q.Push(
                {in_flight_request.initial, (static_cast<uint8_t *>(in_flight_request.current) + cqe_res),
                 in_flight_request.remaining_size - cqe_res, in_flight_request.offset, in_flight_request.fd,
                 reinterpret_cast<std::atomic<IO_STATUS> *>(in_flight_request.result)});
            return;
        }

        // Success
        *result = IO_STATUS::SUCCESS;
        return;
    }

    // Zero progress returns are also considered errors
    switch (-cqe_res) {
    case EIO:
        *result = IO_STATUS::IO_ERROR;
        return;
    case EFAULT:
        *result = IO_STATUS::ACCESS_DENIED;
        return;
    case EPERM:
        *result = IO_STATUS::ACCESS_DENIED;
        return;
    case EBADF:
        *result = IO_STATUS::BAD_FD;
        return;
    case ENOSPC:
        *result = IO_STATUS::OUT_OF_SPACE;
        return;
    default:
        *result = IO_STATUS::OTHER_ERROR;
        return;
    }
}

bool URing::PeekCQAndHandleCQEs() {
    constexpr uint64_t peek_batch_size = 16;
    struct io_uring_cqe *cqes[peek_batch_size];

    uint32_t amount_of_fullfilled_requests = io_uring_peek_batch_cqe(&(this->ring), cqes, peek_batch_size);

    for (int32_t i = 0; i < static_cast<int32_t>(amount_of_fullfilled_requests); i++) {
        uint64_t slot_id = GetSlotIdFromUUID(cqes[i]->user_data);
        IO_OP_CODE op_code = GetOPCodeFromUUID(cqes[i]->user_data);

        switch (op_code) {
        case IO_OP_CODE::READ:
            this->HandleRead(slot_id, cqes[i]->res);
            break;
        case IO_OP_CODE::WRITE:
            this->HandleWrite(slot_id, cqes[i]->res);
            break;
        }
    }

    for (uint32_t i = 0; i < amount_of_fullfilled_requests; i++) {
        io_uring_cqe_seen(&ring, cqes[i]);
    }

    remaining_slots += amount_of_fullfilled_requests;

    return amount_of_fullfilled_requests != 0;
}
