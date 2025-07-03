/*
 * Copyright 2025 The DAPHNE Consortium
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

/*
 * This header contains custom C++ types used to represent the data properties of DAPHNE data objects (e.g., matrices
 * and frames). These types are used in both, the IR/compiler and the runtime.
 */

namespace mlir::daphne {
enum class BoolOrUnknown { Unknown = -1, False = 0, True = 1 };
} // namespace mlir::daphne

// Make it available in the global namespace for backward compatibility if needed
using mlir::daphne::BoolOrUnknown;