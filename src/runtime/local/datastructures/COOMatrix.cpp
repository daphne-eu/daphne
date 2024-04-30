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

#include "COOMatrix.h"


/* TODO COO serializer */
template<typename ValueType>
size_t COOMatrix<ValueType>::serialize(std::vector<char> &buf) const {
    throw std::runtime_error("COOMatrix does not support serialization yet");
//    return DaphneSerializer<COOMatrix<ValueType>>::serialize(this, buf);
}

// explicitly instantiate to satisfy linker
template class COOMatrix<double>;
template class COOMatrix<float>;
template class COOMatrix<int>;
template class COOMatrix<long>;
template class COOMatrix<signed char>;
template class COOMatrix<unsigned char>;
template class COOMatrix<unsigned int>;
template class COOMatrix<unsigned long>;