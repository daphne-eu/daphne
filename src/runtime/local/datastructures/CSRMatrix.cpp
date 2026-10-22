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

#include <runtime/local/io/DaphneSerializer.h>

#include "CSRMatrix.h"

template <typename ValueType> size_t CSRMatrix<ValueType>::serialize(std::vector<char> &buf) const {
    return DaphneSerializer<CSRMatrix<ValueType>>::serialize(this, buf);
}

// explicitly instantiate to satisfy linker
template class CSRMatrix<double>;
template class CSRMatrix<float>;
template class CSRMatrix<int64_t>;
template class CSRMatrix<int32_t>;
template class CSRMatrix<int8_t>;
template class CSRMatrix<uint64_t>;
template class CSRMatrix<uint32_t>;
template class CSRMatrix<uint8_t>;
template class CSRMatrix<bool>;
#if defined(__APPLE__) && defined(__aarch64__)
// On macOS ARM64, size_t is distinct from uint64_t.
template class CSRMatrix<size_t>;
#endif
