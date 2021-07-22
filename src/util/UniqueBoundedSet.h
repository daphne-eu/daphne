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

#ifndef SRC_UTIL_UNIQUEBOUNDEDSET
#define SRC_UTIL_UNIQUEBOUNDEDSET

#include <bits/stdint-uintn.h>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <set>

template <typename QT> class UniqueBoundedSet : std::set<QT> {
public:
  UniqueBoundedSet(size_t K) : K(K){};

    /**
     * @brief Inserts into the set until K unique values are contained.
     * Will replace the biggest value with a smaller inserted value after that.
     */
    void push(const QT &val) {

        if (std::set<QT>::size() < K) {
            std::set<QT>::insert(val);
        } else if (top() > val) {
            pop();
            std::set<QT>::insert(val);
        }
    }

    /**
     * @brief Will remove the biggest value from the set.
     */
    void pop() {
        if (std::set<QT>::empty()) return;

        // erase doesnt support reverse it, so move it back once.
        auto it = std::set<QT>::end();
        std::advance(it, -1);
        std::set<QT>::erase(it);
    }

    /**
     * @brief Returns the biggest value in the set.
     */
    QT top() {
        auto it = std::set<QT>::rbegin();
        return *it;
    }

private:
    size_t K;
};

#endif
