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

#include <api/internal/daphne_internal.h>

extern "C" int daphne(const char* scriptPath) {
    // Assumes that python3 is invoked from the DAPHNE root directory.
    const char * argv[] = {"daphne", "--libdir", "lib", scriptPath};
    int argc = 4;

    return mainInternal(argc, argv);
}