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

#pragma once

#include <ir/daphneir/Daphne.h>
#include <runtime/local/io/FileMetaData.h>

#include <mlir/IR/Value.h>

#include <stdexcept>
#include <string>

namespace CompilerUtils {
    // TODO Copied here from FrameLabelInference, have it just once.
    static std::string getConstantString2(mlir::Value v) {
        if(auto co = llvm::dyn_cast<mlir::daphne::ConstantOp>(v.getDefiningOp()))
            if(auto strAttr = co.value().dyn_cast<mlir::StringAttr>())
                return strAttr.getValue().str();
        throw std::runtime_error(
                "the given value must be a constant of string type"
        );
    }

    [[maybe_unused]] static FileMetaData getFileMetaData(mlir::Value filename) {
        return FileMetaData::ofFile(getConstantString2(filename));
    }

    [[maybe_unused]] static std::string mlirTypeToCppTypeName(mlir::Type t, bool generalizeToStructure = false) {
        if(t.isF64())
            return "double";
        else if(t.isF32())
            return "float";
        else if(t.isSignedInteger(8))
            return "int8_t";
        else if(t.isSignedInteger(32))
            return "int32_t";
        else if(t.isSignedInteger(64))
            return "int64_t";
        else if(t.isUnsignedInteger(8))
            return "uint8_t";
        else if(t.isUnsignedInteger(32))
            return "uint32_t";
        else if(t.isUnsignedInteger(64))
            return "uint64_t";
        else if(t.isSignlessInteger(1))
            return "bool";
        else if(t.isIndex())
            return "size_t";
        else if(auto matTy = t.dyn_cast<mlir::daphne::MatrixType>())
            if(generalizeToStructure)
                return "Structure";
            else {
                switch (matTy.getRepresentation()) {
                case mlir::daphne::MatrixRepresentation::Dense:
                    return "DenseMatrix_" + mlirTypeToCppTypeName(matTy.getElementType(), false);
                case mlir::daphne::MatrixRepresentation::Sparse:
                    return "CSRMatrix_" + mlirTypeToCppTypeName(matTy.getElementType(), false);
                }
            }
        else if(t.isa<mlir::daphne::FrameType>())
            if(generalizeToStructure)
                return "Structure";
            else
                return "Frame";
        else if(t.isa<mlir::daphne::StringType>())
            // This becomes "const char *" (which makes perfect sense for
            // strings) when inserted into the typical "const DT *" template of
            // kernel input parameters.
            return "char";
        else if(t.isa<mlir::daphne::DaphneContextType>())
            return "DaphneContext";
        else if(auto handleTy = t.dyn_cast<mlir::daphne::HandleType>())
            return "Handle_" + mlirTypeToCppTypeName(handleTy.getDataType(), generalizeToStructure);
        else if(t.isa<mlir::daphne::FileType>())
            return "File";
        else if(t.isa<mlir::daphne::DescriptorType>())
            return "Descriptor";
        else if(t.isa<mlir::daphne::TargetType>())
            return "Target";
        throw std::runtime_error(
            "no C++ type name known for the given MLIR type"
        );
    }

    [[maybe_unused]] static bool isMatrixComputation(mlir::Operation *v) {
        return llvm::any_of(v->getOperandTypes(), [&](mlir::Type ty) { return ty.isa<mlir::daphne::MatrixType>(); });
    }
    
    /**
     * @brief Returns the DAPHNE context used in the given function.
     * 
     * Throws if there is not exactly one DAPHNE context.
     * 
     * @param func
     * @return 
     */
    [[maybe_unused]] mlir::Value static getDaphneContext(mlir::FuncOp & func) {
        mlir::Value dctx = nullptr;
        auto ops = func.body().front().getOps<mlir::daphne::CreateDaphneContextOp>();
        for(auto op : ops) {
            if(!dctx)
                dctx = op.getResult();
            else
                throw std::runtime_error(
                        "function body block contains more than one CreateDaphneContextOp"
                );
        }
        if(!dctx)
            throw std::runtime_error(
                    "function body block contains no CreateDaphneContextOp"
            );
        return dctx;
    }
    
    [[maybe_unused]] static bool isObjType(mlir::Type t) {
        return t.isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>();
    }
    
    [[maybe_unused]] static bool hasObjType(mlir::Value v) {
        return isObjType(v.getType());
    }

}