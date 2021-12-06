/*
 *  Copyright 2021 The DAPHNE Consortium
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

#ifndef SRC_COMPILER_COMPILERUTILS_H
#define SRC_COMPILER_COMPILERUTILS_H

#include <ir/daphneir/Daphne.h>
#include <runtime/local/io/FileMetaData.h>

#include <mlir/IR/Value.h>

#include <stdexcept>
#include <string>

struct CompilerUtils {
    // TODO Copied here from FrameLabelInference, have it just once.
    static std::string getConstantString2(mlir::Value v) {
        if(auto co = llvm::dyn_cast<mlir::daphne::ConstantOp>(v.getDefiningOp()))
            if(auto strAttr = co.value().dyn_cast<mlir::StringAttr>())
                return strAttr.getValue().str();
        throw std::runtime_error(
                "the given value must be a constant of string type"
        );
    }
    
    static FileMetaData getFileMetaData(mlir::Value filename) {
        return FileMetaData::ofFile(getConstantString2(filename));
    }
};

#endif //SRC_COMPILER_COMPILERUTILS_H
