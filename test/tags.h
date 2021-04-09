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

#ifndef TEST_TAGS_H
#define TEST_TAGS_H

// The following tags are intended to be used as the second argument to the
// TEST_CASE macros of catch2. You can easily combine tags by using multiple
// tag macros separated by whitespace, e.g., if TAG_A is "[a]" and TAG_B is
// "[b]", then TAG_A TAG_B is "[a]" "[b]", which is equivalent to "[a][b]".

#define TAG_DATASTRUCTURES "[datastructures]"
#define TAG_KERNELS "[kernels]"

#endif //TEST_TAGS_H

