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
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

template<typename T>
struct ThreadSafeStack {
    T *data;
    std::atomic<uint64_t> size = 0;
    uint64_t capacity;
    static constexpr uint64_t default_initial_capacity = 16;
    std::mutex lck;

    // Assumes lock is currently held
    void Grow() {
        T *new_buffer = static_cast<T *>(std::malloc(sizeof(T) * capacity * 2));
        std::memcpy(new_buffer, data, sizeof(T) * size);
        free(data);
        data = new_buffer;
        capacity *= 2;
    }

    void Push(T task) {
        lck.lock();

        uint64_t current_size = size.load();
        if (current_size >= capacity) {
            Grow();
        }
        data[current_size] = task;
        size++;

        lck.unlock();
    }

    void Push(const std::vector<T> &tasks) {
        lck.lock();

        uint64_t current_size = size.load();
        while (current_size + tasks.size() > capacity) {
            Grow();
        }
        for (uint64_t i = 0; i < tasks.size(); i++) {
            data[current_size + i] = tasks[i];
        }
        size += tasks.size();

        lck.unlock();
    }

    std::optional<T> TryPop() {
        lck.lock();

        if (size == 0) {
            lck.unlock();
            return std::nullopt;
        }

        T result = data[--size];

        lck.unlock();

        return result;
    }

    std::vector<T> TryPop(uint64_t max_elements_poped) {
        lck.lock();

        if (size == 0) {
            lck.unlock();
            return {};
        }

        uint64_t elements_to_pop = size.load() > max_elements_poped ? max_elements_poped : size.load();

        std::vector<T> results;
        results.resize(elements_to_pop);
        for (uint64_t i = 0; i < elements_to_pop; i++) {
            results[i] = data[size - 1 - i];
        }
        size -= elements_to_pop;

        lck.unlock();

        return results;
    }

    std::optional<T> PollThenTryPop() {
        if (size == 0) {
            return std::nullopt;
        }

        lck.lock();

        if (size == 0) {
            lck.unlock();
            return std::nullopt;
        }

        T result = data[--size];

        lck.unlock();

        return result;
    }

    // Does not acquire the lock -> likely only useful for e.g. polling style operations 
    uint64_t Size() {
        return size.load();
    }

    // Assumes id is in bounds
    template<bool allready_holding_lck>
    void Erase(uint64_t id) {
        if constexpr (!allready_holding_lck) {
            lck.lock();
        }
        T *new_buffer                  = static_cast<T *>(std::malloc(sizeof(T) * capacity));
        uint64_t size_before_decrement = size--;
        std::memcpy(new_buffer, data, sizeof(T) * id);
        std::memcpy(new_buffer + id, data + id + 1, sizeof(T) * (size_before_decrement - id - 1));
        free(data);
        data = new_buffer;
        if constexpr (!allready_holding_lck) {
            lck.unlock();
        }
    }

    ThreadSafeStack(uint64_t initial_capacity = default_initial_capacity) {
        data = static_cast<T *>(std::malloc(sizeof(T) * initial_capacity));
    }

    ~ThreadSafeStack() {
        std::free(data);
    }
};

// Fixed buffer, random access Container / Allocator
template<typename DataType>
struct Pool {
    std::unique_ptr<DataType[]> data;
    uint64_t max_size;
    std::unique_ptr<std::mutex[]> entry_lcks;
    // For polling; May not be accurate
    std::unique_ptr<std::atomic<bool>[]> is_in_use;
    std::atomic<uint64_t> currently_occupied_slots = 0;

    Pool(uint64_t max_size) : max_size(max_size) {
        data       = std::make_unique<DataType[]>(max_size);
        is_in_use  = std::make_unique<std::atomic<bool>[]>(max_size);
        entry_lcks = std::make_unique<std::mutex[]>(max_size);
        for (uint64_t i = 0; i < max_size; i++) {
            is_in_use[i] = false;
        }
    };

    template<bool keep_holding_lock_after_success>
    std::optional<uint64_t> Alloc() {
        for (uint64_t i = 0; i < max_size; i++) {
            entry_lcks[i].lock();
            if (!is_in_use[i]) {
                is_in_use[i] = true;
                currently_occupied_slots++;
                if constexpr (!keep_holding_lock_after_success) {
                    entry_lcks[i].unlock();
                }
                return i;
            }
            entry_lcks[i].unlock();
        }

        return std::nullopt;
    }

    template<bool already_holding_lock>
    void Free(uint64_t id) {
        if constexpr (!already_holding_lock) {
            entry_lcks[id].lock();
        }
        if (is_in_use[id]) {
            currently_occupied_slots--;
            is_in_use[id] = false;
        }
        entry_lcks[id].unlock();
    }

    // Attempts to insert item into pool and returns the according index.
    // Fails if full, but also may spuriously fail if polling flag is set
    template<bool poll_before_acquire>
    std::optional<uint64_t> TryInsert(DataType to_insert) {
        for (uint64_t i = 0; i < max_size; i++) {
            if constexpr (poll_before_acquire) {
                if (!is_in_use[i]) {
                    entry_lcks[i].lock();

                    if (!is_in_use[i]) {
                        is_in_use[i] = true;
                        data[i]      = to_insert;
                        currently_occupied_slots++;
                        entry_lcks[i].unlock();
                        return i;
                    }
                    entry_lcks[i].unlock();
                }
            } else {
                entry_lcks[i].lock();
                if (!is_in_use[i]) {
                    is_in_use[i] = true;
                    data[i]      = to_insert;
                    currently_occupied_slots++;
                    entry_lcks[i].unlock();
                    return i;
                }
                entry_lcks[i].unlock();
            }
        }
        return std::nullopt;
    }

    // Like Try equivalent but blocks until successful
    uint64_t Insert(DataType to_insert) {
        auto possible_result = TryInsert<true>(to_insert);
        if (possible_result) {
            return possible_result.value();
        }

        while (true) {
            possible_result = TryInsert<false>(to_insert);
            if (possible_result) {
                return possible_result.value();
            }
        }
    }

    std::optional<DataType> GetFirst() {
        for (uint64_t i = 0; i < max_size; i++) {
            entry_lcks[i].lock();

            if (is_in_use[i]) {
                is_in_use[i] = false;
                DataType tmp = data[i];
                entry_lcks[i].unlock();
                return tmp;
            }
            entry_lcks[i].unlock();
        }

        return std::nullopt;
    }

    template<bool keep_holding_lock_after_success>
    std::optional<uint64_t> Find(DataType to_find) {
        for (uint64_t i = 0; i < max_size; i++) {
            if (is_in_use[i]) {
                entry_lcks[i].lock();

                if (is_in_use[i]) {
                    if (data[i] == to_find) {
                        if constexpr (!keep_holding_lock_after_success) {
                            entry_lcks[i].unlock();
                        }
                        return i;
                    }
                }
                entry_lcks[i].unlock();
            }
        }
        // No match
        return std::nullopt;
    }

    // For use with T==DataType or with other T that has a matching == operator
    template<typename T>
    bool FindAndRemove(T to_find) {
        for (uint64_t i = 0; i < max_size; i++) {
            entry_lcks[i].lock();

            if (is_in_use[i]) {
                if (data[i] == to_find) {
                    // Found a match
                    is_in_use[i] = false;
                    currently_occupied_slots--;
                    entry_lcks[i].unlock();
                    return true;
                }
            }
            entry_lcks[i].unlock();
        }
        // No match
        return false;
    }

    // For use with T==DataType or with other T that has a matching == operator
    template<typename T>
    std::optional<DataType> FindAndExtract(T to_find) {
        for (uint64_t i = 0; i < max_size; i++) {
            entry_lcks[i].lock();

            if (is_in_use[i]) {
                if (data[i] == to_find) {
                    // Found a match
                    is_in_use[i] = false;
                    currently_occupied_slots--;
                    DataType result = data[i];
                    entry_lcks[i].unlock();
                    return result;
                }
            }
            entry_lcks[i].unlock();
        }
        // No match
        return std::nullopt;
    }

    void LockAllSlots() {
        for (uint64_t i = 0; i < max_size; i++) {
            entry_lcks[i].lock();
        }
    }

    void UnlockAllSlots() {
        for (uint64_t i = 0; i < max_size; i++) {
            entry_lcks[i].unlock();
        }
    }

    DataType Extract(uint64_t id) {
        entry_lcks[id].lock();
        DataType tmp  = data[id];
        is_in_use[id] = false;
        currently_occupied_slots--;
        entry_lcks[id].unlock();
        return tmp;
    }
};
