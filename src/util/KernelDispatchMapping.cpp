/*
 * Copyright 2024 The DAPHNE Consortium
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

#include "KernelDispatchMapping.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/BuiltinOps.h>

KernelDispatchMapping &KernelDispatchMapping::instance() {
    static KernelDispatchMapping INSTANCE;
    return INSTANCE;
}

int KernelDispatchMapping::registerKernel(std::string name,
                                          mlir::Operation *op) {
    std::lock_guard<std::mutex> lg(m_dispatchMapping);
    int kId = kIdCounter++;
    if (auto flcLoc = llvm::dyn_cast<mlir::FileLineColLoc>(op->getLoc())) {
        auto fName = flcLoc.getFilename().str();
        dispatchMapping[kId] = {name, fName, flcLoc.getLine(),
                                  flcLoc.getColumn()};
        currentLoc = flcLoc;
    } else {
        dispatchMapping[kId] = {name, currentLoc.getFilename().str(),
                                  currentLoc.getLine(), currentLoc.getColumn()};
    }
    return kId;
}

KDMInfo KernelDispatchMapping::getKernelDispatchInfo(int kId) {
    std::lock_guard<std::mutex> lg(m_dispatchMapping);
    return dispatchMapping.at(kId);
}
