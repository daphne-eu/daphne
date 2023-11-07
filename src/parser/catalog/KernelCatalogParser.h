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

#pragma once

#include <compiler/catalog/KernelCatalog.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief A parser for kernel information.
 */
class KernelCatalogParser {

    /**
     * @brief A mapping from C++ type name strings to MLIR types used for parsing input/output types of kernels.
     */
    std::unordered_map<std::string, mlir::Type> typeMap;

    /**
     * @brief Maps the given C++ type names to MLIR types.
     * 
     * @param in The vector of C++ type names.
     * @param out The vector of corresponding MLIR types.
     * @param word Typically either `"argument"` or `"result"`.
     * @param kernelFuncName The name of the kernel function for which this method is called (for error message).
     * @param opMnemonic The mnemonic of the operation for which this method is called (for error message).
     * @param backend The backend for which this method is called (for error message).
     */
    void mapTypes(
        const std::vector<std::string> & in,
        std::vector<mlir::Type> & out,
        const std::string & word,
        const std::string & kernelFuncName,
        const std::string & opMnemonic,
        const std::string & backend
    ) const;

public:

    /**
     * @brief Creates a new kernel catalog parser.
     */
    KernelCatalogParser(mlir::MLIRContext * mctx);

    /**
     * @brief Parses kernel information from the given file and registers them with the given kernel catalog.
     * 
     * @param filePath The path to the file to extract kernel information from.
     * @param kc The kernel catalog to register the kernels with.
     */
    void parseKernelCatalog(const std::string & filePath, KernelCatalog & kc) const;
};