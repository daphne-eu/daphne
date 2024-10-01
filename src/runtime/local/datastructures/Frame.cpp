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

#include <ostream>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/DaphneSerializer.h>

std::ostream &operator<<(std::ostream &os, const Frame &obj) {
    obj.print(os);
    return os;
}

size_t Frame::serialize(std::vector<char> &buf) const { return DaphneSerializer<Frame>::serialize(this, buf); }