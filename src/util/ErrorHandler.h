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

#include "KernelDispatchMapping.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Visitors.h"
#include <fstream>
#include <system_error>
#include <vector>

struct ErrorHandler {
    /**
     * Note: We return an exception than throwing it for the
     * following reason: If the functions threw the exception, we would use the
     * functions like a replacement for a C++ throw statement. However, that
     * would be hard to understand for the C++ compiler in some cases. For
     * instance, assume a function f() with non-void return value contains only
     * an if-then-else statement, whose then-branch returns a value and whose
     * else-branch throws an exception. The C++ compiler could complain about
     * reaching the end of control flow without a return or throw statement in
     * f(). By returning the exception, the caller can simply throw it as in
     * `throw ErrorHandler::compilerError(...);`, and that will look fine for
     * the C++ compiler.
     */
  private:
  public:
    /*
     * Used to create an exception for precompiled kernels.
     * \param kId KernelID obtained from calling
     * KernelDispatchMapping::registerKernel. Is passed to all kernel calls.
     * \param kdm KernelDispatchMapping instance held by the DaphneContext.
     */
    static std::runtime_error runtimeError(int kId, std::string msg,
                                           KernelDispatchMapping *kdm);

    /*
     * To be used during compilation, emits the passed msg and provides
     * a hint in the source code location.
     * \param loc Holds information about the source code location of the op
     * that the action processed.
     * \param action Name or description of pass or other action that has
     * failed.
     * \param msg User-facing error message.
     */
    static std::runtime_error compilerError(mlir::Location loc,
                                            const std::string &action,
                                            const std::string &msg);

    /*
     * To be used during compilation, emits the passed msg and provides
     * a hint in the source code location.
     * \param op Points to the operation
     * that the action processed.
     * \param action Name or description of pass or other action that has
     * failed.
     * \param msg User-facing error message.
     */
    static std::runtime_error compilerError(mlir::Operation *op,
                                            const std::string &action,
                                            const std::string &msg);

    /*
     * To be used when catching an exception and rethrowing the exception.
     * Allows to add additional stacktrace informations through the action
     * param.
     * \param action Name or description of pass or other action that has
     * failed.
     * \param msg Recommended to reuse the message from the caught exception,
     * but additional information can be added to the string.
     */
    static std::runtime_error rethrowError(const std::string &action,
                                           const std::string &msg);

    /*
     * Creates an std::runtime_error instance with a header, an error message,
     * and a hint referencing the source code line containing the error.
     *
     * \param header Should contain the origin of the exception, e.g., the
     * responsible pass or kernel symbol.
     * \param msg The msg to the user explaining what failed (and why, when
     * possible).
     * \param file The source file, e.g., "test.daphne"
     * \param line The line in the source code from which the error originated.
     * \param col The column position in the source code at the given line from
     * which the error originated.
     *
     * Note: the [error] prefix in the example is added in our root catch-block
     * in daphne_internals.cpp
     *
     * Example output:
     *
     *    [error]: HEADER [ MSG ]
     *       | Source file -> FILE:LINE:COL
     *       |
     *  LINE | SOURCE
     *       | ^~~
     *       |
     *       |
     */
    static std::runtime_error makeError(std::string header, std::string msg,
                                 std::string file, unsigned int line,
                                 unsigned int col);

    /*
     * Writes the current module IR to the file "module_fail.log" on disk.
     */
    static void dumpModuleToDisk(mlir::ModuleOp &module);
};
