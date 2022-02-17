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

#pragma once

#if defined(__clang__)
#define PRAGMA_DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#define PRAGMA_DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#define PRAGMA_LOOP_VECTORIZE _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define PRAGMA_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define PRAGMA_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define PRAGMA_LOOP_VECTORIZE _Pragma("GCC ivdep")
#else
#define PRAGMA_DIAGNOSTIC_PUSH
#endif

#if defined(__JETBRAINS_IDE__)
#define PRAGMA_CLION_IGNORED(x) _Pragma("ide diagnostic ignored \"" x "\"")
PRAGMA_CLION_IGNORED("UnusedImportStatement")
#else
#define PRAGMA_CLION_IGNORED
#endif

