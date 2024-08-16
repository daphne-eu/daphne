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

#pragma once

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

/**
 * Data-object keeping source file location information for a specific kernel
 * dispatch.
 */
struct KDMInfo {
    std::string kernelName{};
    std::string fileName{};
    unsigned int line{};
    unsigned int column{};
};

/**
 * Singleton class that holds the mapping from calls to the kernel library.
 * These mappings are registered during the lowering pipeline of the compiler,
 * and maps a kernel identifier (int) to an instance of KDMInfo.
 */
struct KernelDispatchMapping {
   private:
    int kIdCounter{0};
    std::mutex m_dispatchMapping{};
    std::unordered_map<int, KDMInfo> dispatchMapping{};
    /**
     * Used as fallback for OPs without a source location, e.g., for calls to
     * the decRef kernel.
     */
    mlir::FileLineColLoc currentLoc{};

   public:
    std::unordered_map<int, KDMInfo>::iterator begin() {
        return dispatchMapping.begin();
    }
    std::unordered_map<int, KDMInfo>::iterator end() {
        return dispatchMapping.end();
    }
    std::unordered_map<int, KDMInfo>::const_iterator begin() const {
        return dispatchMapping.begin();
    }
    std::unordered_map<int, KDMInfo>::const_iterator end() const {
        return dispatchMapping.end();
    }

    static KernelDispatchMapping& instance();

    /**
     * Used to register kernel call during source code lowering.
     * \param name The symbol name of the kernel.
     * \param op The mlir::Operation being lowered to dispatch a kernel call.
     */
    int registerKernel(std::string name, mlir::Operation* op);
    //
    KDMInfo getKernelDispatchInfo(int kId);
};
