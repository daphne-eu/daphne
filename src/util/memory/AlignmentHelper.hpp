/*
 * Copyright 2023 The DAPHNE Consortium
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

#ifndef SRC_UTIL_MEMORY_ALIGNMENTHELPER_HPP
#define SRC_UTIL_MEMORY_ALIGNMENTHELPER_HPP

#include <cstddef>

struct AlignmentHelper {

    class Alignment {
      private:
        const void *ptr;

        size_t alignment;
        size_t offset;


      public:
        Alignment(const void *ptr, size_t alignment) : ptr(ptr) {
            this->alignment = alignment;
            this->offset = reinterpret_cast<size_t>(ptr) % alignment;
        }

        bool isAligned() const {
            return offset == 0;
        }

        size_t getOffset() const {
            return offset;
        }

        size_t getAlignment() const {
            return alignment;
        }

        const void *getPtr() const {
            return ptr;
        }

        const void *getFirstAlignedPtrWithin() const {
            size_t ptrValue = reinterpret_cast<size_t>(ptr);
            return reinterpret_cast<const void *>(ptrValue + (ptrValue % alignment));
        }

        bool operator==(const Alignment &other) const {
            return offset == other.offset;
        }

    };


    static const Alignment getAlignment(const void *ptr, size_t alignment) {
        return Alignment(ptr, alignment);
    }

};



#endif //SRC_UTIL_MEMORY_ALIGNMENTHELPER_HPP


