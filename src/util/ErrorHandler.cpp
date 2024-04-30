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

#include "ErrorHandler.h"
#include "KernelDispatchMapping.h"
#include <filesystem>

#include <ir/daphneir/Daphne.h>
#include <mlir/IR/Location.h>
#include <stdexcept>

static constexpr auto BREADCRUMB = " -> ";
static constexpr auto INDENT = "   ";
static constexpr auto DAPHNE_RED = "\e[38;2;247;1;70m";
static constexpr auto DAPHNE_BLUE = "\e[38;2;120;137;251m";
static constexpr auto RESET_COLOR = "\x1b[0m";

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
std::runtime_error ErrorHandler::makeError(std::string header, std::string msg,
                             std::string file, unsigned int line,
                             unsigned int col) {
    std::stringstream s;
    s << header;
    std::filesystem::path p = file;
    s << INDENT << DAPHNE_BLUE << " | " << RESET_COLOR << "Source file -> "
      << std::filesystem::absolute(p) << ':' << line << ':' << col << "\n";

    auto fStream = std::ifstream(file);
    s << INDENT << DAPHNE_BLUE << " | " << RESET_COLOR << "\n";

    std::string l;
    for (unsigned int i = 1; i <= line; i++)
        std::getline(fStream, l);

    s << DAPHNE_BLUE << std::setw(3) << line << " | " << RESET_COLOR << l;
    std::string hint = std::string(col, ' ') + "^" + std::string(2, '~');
    s << "\n"
      << INDENT << DAPHNE_BLUE << " | " << DAPHNE_RED << hint << RESET_COLOR
      << "\n\n";

    return std::runtime_error(s.str());
}

static constexpr int UNREGISTERED = -1;
std::runtime_error ErrorHandler::runtimeError(int kId, std::string msg,
                                              KernelDispatchMapping *kdm) {
    if (kId == UNREGISTERED)
        return std::runtime_error(msg);

    auto kdi = kdm->getKernelDispatchInfo(kId);
    std::string header =
        std::string("The kernel-function ") + DAPHNE_BLUE + kdi.kernelName +
        RESET_COLOR + " failed during runtime with the following message [ " +
        DAPHNE_RED + msg + RESET_COLOR + " ]\n";

    return makeError(header, msg, kdi.fileName, kdi.line, kdi.column);
}

std::runtime_error ErrorHandler::compilerError(mlir::Operation *op,
                                               const std::string &pass,
                                               const std::string &msg) {
    return compilerError(op->getLoc(), pass, msg);
}

std::runtime_error ErrorHandler::compilerError(mlir::Location loc,
                                               const std::string &pass,
                                               const std::string &msg) {

    auto flcLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc);
    std::stringstream header;
    auto fName = flcLoc.getFilename().str();
    header << DAPHNE_BLUE << pass << RESET_COLOR
           << " failed with the following message [ " << DAPHNE_RED << msg
           << RESET_COLOR << " ]\n";
    return makeError(header.str(), msg, fName, flcLoc.getLine(),
                     flcLoc.getColumn());
}

std::runtime_error ErrorHandler::rethrowError(const std::string &action,
                                              const std::string &msg) {
    std::stringstream s;
    s << action << BREADCRUMB << msg;
    return std::runtime_error(s.str());
}

void ErrorHandler::dumpModuleToDisk(mlir::ModuleOp &module) {
    std::string fName = "module_fail.log";
    std::error_code EC;
    llvm::raw_fd_ostream fs(fName, EC);
    module->print(fs);
}
