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

#include "CSCMatrix.h"

template<typename ValueType>
size_t CSCMatrix<ValueType>::serialize(std::vector<char> &buf) const {
    return DaphneSerializer<CSCMatrix<ValueType>>::serialize(this, buf);
}

// explicitly instantiate to satisfy linker
template class CSCMatrix<double>;
template class CSCMatrix<float>;
template class CSCMatrix<int>;
template class CSCMatrix<long>;
template class CSCMatrix<signed char>;
template class CSCMatrix<unsigned char>;
template class CSCMatrix<unsigned int>;
template class CSCMatrix<unsigned long>;
