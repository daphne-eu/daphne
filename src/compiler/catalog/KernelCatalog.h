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

#pragma once

#include <compiler/utils/TypePrinting.h>

#include <mlir/IR/Types.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief Stores information on a single kernel.
 */
struct KernelInfo {
    /**
     * @brief The name of the pre-compiled kernel function.
     */
    const std::string kernelFuncName;

    // TODO Add the path to the shared library containing the kernel function.

    /**
     * @brief The kernel's result types.
     */
    const std::vector<mlir::Type> resTypes;

    /**
     * @brief The kernel's argument types.
     */
    const std::vector<mlir::Type> argTypes;

    // TODO Maybe unify this with ALLOCATION_TYPE.
    /**
     * @brief The targeted backend (e.g., hardware accelerator).
     */
    const std::string backend;

    /**
     * @brief The path to the shared library containing the pre-compiled kernel.
     *
     * This path can be absolute or relative to the present working directory.
     */
    const std::string libPath;

    KernelInfo(
        const std::string kernelFuncName,
        const std::vector<mlir::Type> resTypes,
        const std::vector<mlir::Type> argTypes,
        const std::string backend,
        const std::string libPath
    ) :
        kernelFuncName(kernelFuncName), resTypes(resTypes), argTypes(argTypes), backend(backend), libPath(libPath)
    {
        //
    }
};

/**
 * @brief Stores information on kernels registered in the DAPHNE compiler.
 */
class KernelCatalog {
    /**
     * @brief The central data structure mapping DaphneIR operations to registered kernels.
     * 
     * The DaphneIR operation is represented by its mnemonic. The kernels are represented
     * by their kernel information.
     */
    std::unordered_map<std::string, std::vector<KernelInfo>> kernelInfosByOp;

    /**
     * @brief Prints the given kernel information.
     * 
     * @param opMnemonic The mnemonic of the corresponding DaphneIR operation.
     * @param kernelInfos The kernel information to print.
     * @param os The stream to print to. Defaults to `std::cerr`.
     */
    void dumpKernelInfos(const std::string & opMnemonic, const std::vector<KernelInfo> & kernelInfos, std::ostream & os = std::cerr) const {
        os << "- operation `" << opMnemonic << "` (" << kernelInfos.size() << " kernels)" << std::endl;
        for(KernelInfo ki : kernelInfos) {
            os << "  - kernel `" << ki.kernelFuncName << "`: (";
            for(size_t i = 0; i < ki.argTypes.size(); i++) {
                os << ki.argTypes[i];
                if(i < ki.argTypes.size() - 1)
                    os << ", ";
            }
            os << ") -> (";
            for(size_t i = 0; i < ki.resTypes.size(); i++) {
                os << ki.resTypes[i];
                if(i < ki.resTypes.size() - 1)
                    os << ", ";
            }
            os << ") for backend `" << ki.backend  << "` (in `" << ki.libPath << "`)" << std::endl;
        }
    }

public:
    /**
     * @brief Registers the given kernel information as a kernel for the DaphneIR
     * operation with the given mnemonic.
     * 
     * @param opMnemonic The DaphneIR operation's mnemonic.
     * @param kernelInfo The information on the kernel.
     */
    void registerKernel(std::string opMnemonic, KernelInfo kernelInfo) {
        kernelInfosByOp[opMnemonic].push_back(kernelInfo);
    }

    /**
     * @brief Retrieves information on all kernels registered for the given DaphneIR operation.
     * 
     * @param opMnemonic The mnemonic of the DaphneIR operation.
     * @return A vector of kernel information, or an empty vector if no kernels are registered
     * for the given operation.
     */
    const std::vector<KernelInfo> getKernelInfos(const std::string & opMnemonic) const {
        auto it = kernelInfosByOp.find(opMnemonic);
        if(it != kernelInfosByOp.end())
            return it->second;
        else
            return {};
    }

    /**
     * @brief Returns the mnemonic of the operation for which a kernel with the
     * given name is registered, or throws an exception if there is none.
     *
     * @param kernelFuncName The name of the kernel function to look for.
     * @return The mnemonic of the operation.
     */
    std::string getOpMnemonic(const std::string & kernelFuncName) {
        for(auto it : kernelInfosByOp) {
            std::string opMnemonic = it.first;
            const std::vector<KernelInfo> & kis = it.second;
            for(auto it2 : kis)
                if(it2.kernelFuncName == kernelFuncName)
                    return opMnemonic;
        }
        throw std::runtime_error(
            "no kernel with name `" + kernelFuncName + "` registered in the kernel catalog"
        );
    }

    /**
     * @brief Prints high-level statistics on the kernel catalog.
     * 
     * @param os The stream to print to. Defaults to `std::cerr`.
     */
    void stats(std::ostream & os = std::cerr) const {
        const size_t numOps = kernelInfosByOp.size();
        size_t numKernels = 0;
        for(auto it = kernelInfosByOp.begin(); it != kernelInfosByOp.end(); it++)
            numKernels += it->second.size();
        os << "KernelCatalog (" << numOps << " ops, " << numKernels << " kernels)" << std::endl;
    }

    /**
     * @brief Prints this kernel catalog.
     * 
     * @param opMnemonic If an empty string, print registered kernels for all DaphneIR
     * operations; otherwise, consider only the specified DaphneIR operation.
     * @param os The stream to print to. Defaults to `std::cerr`.
     */
    void dump(std::string opMnemonic = "", std::ostream & os = std::cerr) const {
        stats(os);
        if(opMnemonic.empty())
            // Print info on all ops.
            for(auto it = kernelInfosByOp.begin(); it != kernelInfosByOp.end(); it++)
                dumpKernelInfos(it->first, it->second, os);
        else
            // Print info on specified op only.
            dumpKernelInfos(opMnemonic, getKernelInfos(opMnemonic), os);
    }

    /**
     * @brief Returns all distinct kernel libraries in the form of a mapping from
     * the library path to the constant `false`.
     *
     * @return A mapping from each distict kernel library path to the constant `false`.
     */
    std::unordered_map<std::string, bool> getLibPaths() const {
        std::unordered_map<std::string, bool> res;

        for(auto it : kernelInfosByOp) {
            const std::vector<KernelInfo> & kis = it.second;
            for(auto it2 : kis)
                res[it2.libPath] = false;
        }

        return res;
    }
};