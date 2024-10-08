/*
 *  Copyright 2023 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <compiler/utils/TypePrinting.h>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Types.h>

#include <ostream>
#include <string>

std::ostream &operator<<(std::ostream &os, mlir::Type t) {
    std::string s;
    llvm::raw_string_ostream rsos(s);
    t.print(rsos);
    os << s;
    return os;
}