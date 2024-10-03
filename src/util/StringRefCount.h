/*
 * Copyright 2024 The DAPHNE Consortium
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

#include <spdlog/spdlog.h>

#include <map>
#include <memory>

class StringRefCounter {

    std::shared_ptr<spdlog::logger> logger;

    // This map keeps track of string allocations to be able to remove them when
    // they are not needed anymore
    std::map<uintptr_t, size_t> stringRefCount;
    std::mutex mtxStrRefCnt;

  public:
    StringRefCounter() { logger = spdlog::get("runtime"); }

    ~StringRefCounter() {
        if (!stringRefCount.empty()) {
            // This should not happen.
            logger->warn("{} string refs still present while destroying "
                         "StringRefCounter - this should not happen.",
                         stringRefCount.size());
        }
    }

    /**
     * @brief Increases the reference counter of the given string.
     *
     * If no reference counter is stored for this string, a prior value of 1 is
     * implicitly assumed, i.e., then the reference counter is increased to an
     * explicitly stored value of 2.
     *
     * @param arg The string that is to be tracked.
     *
     */
    void inc(const char *arg);

    /**
     * @brief Decreases the reference counter of the given string.
     *
     * If no reference counter is stored for this string, a prior value of 1 is
     * implicitly assumed, i.e., then no changes are made to the stored
     * reference counters and `false` is returned.
     *
     * @param arg The string that is to be tracked.
     *
     * @return `false` if the reference counter became zero through the
     * decrement, `true` otherwise.
     */
    bool dec(const char *arg);

    static StringRefCounter &instance();
};
