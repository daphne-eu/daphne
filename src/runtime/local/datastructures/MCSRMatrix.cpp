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

#include "MCSRMatrix.h"

template<typename ValueType>
size_t MCSRMatrix<ValueType>::serialize(std::vector<char> &buf) const {
    return DaphneSerializer<MCSRMatrix<ValueType>>::serialize(this, buf);
}

// explicitly instantiate to satisfy linker
template class MCSRMatrix<double>;
template class MCSRMatrix<float>;
template class MCSRMatrix<int>;
template class MCSRMatrix<long>;
template class MCSRMatrix<signed char>;
template class MCSRMatrix<unsigned char>;
template class MCSRMatrix<unsigned int>;
template class MCSRMatrix<unsigned long>;
