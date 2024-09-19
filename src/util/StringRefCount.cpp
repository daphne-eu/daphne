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

#include "StringRefCount.h"

void StringRefCounter::inc(const char *arg) {
    auto ptr = reinterpret_cast<uintptr_t>(arg);
    const std::lock_guard<std::mutex> lock(mtxStrRefCnt);
    if (auto found = stringRefCount.find(ptr); found != stringRefCount.end()) {
        // If the string was found, increase its reference counter.
        found->second++;
        logger->debug(
            "StringRefCounter::inc: ptr={}; arg={}; found and incremented", ptr,
            arg);
    } else {
        // If the string was not found, implicitly assume a prior counter of 1,
        // and increase the counter to 2.
        stringRefCount.insert({ptr, 2});
        logger->debug(
            "StringRefCounter::inc: ptr={}; arg={}; not found and set to 2",
            ptr, arg);
    }
}

bool StringRefCounter::dec(const char *arg) {
    auto ptr = reinterpret_cast<uintptr_t>(arg);
    const std::lock_guard<std::mutex> lock(mtxStrRefCnt);
    if (auto found = stringRefCount.find(ptr); found != stringRefCount.end()) {
        // If the string was found, decrease its reference counter.
        found->second--;
        logger->debug(
            "StringRefCounter::dec: ptr={}; arg={}; found and decremented", ptr,
            arg);
        if (found->second == 0) {
            // If the reference counter became zero, erase it and return false.
            logger->debug(
                "StringRefCounter::dec: ptr={}; arg={}; became zero and erased",
                ptr, arg);
            stringRefCount.erase(found);
            return false;
        }
        // If the reference counter did not become zero, keep it and return
        // true.
        return true;
    } else {
        // If the string was not found, implicitly assume a prior counter of 1,
        // don't change the stored counters, just return false.
        logger->debug("StringRefCounter::dec: ptr={}; arg={}; not found", ptr,
                      arg);
        return false;
    }
}

StringRefCounter &StringRefCounter::instance() {
    static StringRefCounter INSTANCE;
    return INSTANCE;
}