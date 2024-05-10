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
#include <parser/metadata/MetaDataParser.h>
#include "util/ErrorHandler.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <stdexcept>
#include <string>

struct CompilerUtils {

private:

    template<typename ValT, typename AttrT>
    static std::pair<bool, ValT> isConstantHelper(mlir::Value v, const std::function<ValT(const AttrT &)>& func) {
        if(auto co = v.getDefiningOp<mlir::daphne::ConstantOp>())
            if(auto attr = co.getValue().dyn_cast<AttrT>())
                return std::make_pair(true, func(attr));
        if(auto co = v.getDefiningOp<mlir::arith::ConstantOp>())
            if(auto attr = co.getValue().dyn_cast<AttrT>())
                return std::make_pair(true, func(attr));
        return std::make_pair(false, ValT(0));
    }

    template<typename ValT, typename AttrT>
    static ValT constantOrThrowHelper(mlir::Value v, std::function<ValT(const AttrT &)> func, const std::string & errorMsg, const std::string & valTypeName) {
        auto p = isConstantHelper<ValT, AttrT>(v, func);
        if(p.first)
            return p.second;
        else
            throw ErrorHandler::compilerError(v.getLoc(), "constantOrThrow",
                    errorMsg.empty() ?
                    ("the given value must be a constant of " + valTypeName + " type")
                    : errorMsg
            );
    }

    template<typename ValT, typename AttrT>
    static ValT constantOrDefaultHelper(mlir::Value v, ValT d, std::function<ValT(const AttrT &)> func) {
        auto p = isConstantHelper<ValT, AttrT>(v, func);
        if(p.first)
            return p.second;
        else
            return d;
    }
    
public:

    /**
     * @brief If the given `Value` is defined by some constant operation, return that constant
     * operation; otherwise, return `nullptr`.
     * 
     * @param v The `Value`.
     * @return The defining constant operation or `nullptr`.
     */
    static mlir::Operation * constantOfAnyType(mlir::Value v) {
        if(auto co = v.getDefiningOp<mlir::daphne::ConstantOp>())
            return co;
        if(auto co = v.getDefiningOp<mlir::arith::ConstantOp>())
            return co;
        return nullptr;
    }

    /**
     * @brief Returns if the given `Value` is a constant, and if so, also the constant itself.
     * 
     * @tparam T The C++ type of the constant to extract.
     * @param v The `Value`.
     * @return If the given value is a constant: a pair of the value `true` and the constant value as type `T`;
     * otherwise, a pair of the value `false` and an unspecified value of type `T`.
     */
    template<typename T>
    static std::pair<bool, T> isConstant(mlir::Value v);
    
    /**
     * @brief Returns a constant extracted from the given `Value`, or throws an exception if this is not possible.
     * 
     * @tparam T The C++ type of the constant to extract.
     * @param v The `Value`.
     * @param errorMsg The message of the exception to throw. In case of an empty string (default), the exception
     * will have a generic error message.
     * @return The extracted constant as a value of type `T`, if the given value is a constant.
     */
    template<typename T>
    static T constantOrThrow(mlir::Value v, const std::string & errorMsg = "");

    /**
     * @brief Returns a constant extracted from the given `Value`, or a default value if this is not possible.
     * 
     * @tparam T The C++ type of the constant to extract.
     * @param v The `Value`.
     * @param d The default value.
     * @return The extracted constant as a value of type `T`, if the given value is a constant, or the given
     * default value, otherwise.
     */
    template<typename T>
    static T constantOrDefault(mlir::Value v, T d);

    [[maybe_unused]] static FileMetaData getFileMetaData(mlir::Value filename);

    /**
     * @brief Produces a string containing the C++ type name of the corresponding MLIR type. Mainly used to
     * generate function names for generated kernel libraries. This function is defined recursively to also print
     * the value types of templated containers (e.g., DenseMatrix<float>). A pragma is added to silence clang-tidy which
     * might complain about recursion.
     *
     * @param t MLIR type name
     * @param angleBrackets If `true` (default), angle brackets are used for C++ template types (e.g., `DenseMatrix<float>`);
     * Otherwise, underscores are used (e.g., `DenseMatrix_float`).
     * @param generalizeToStructure If `true`, `Structure` is used instead of derived types like `DenseMatrix` etc.
     * @return A string representation of the C++ type names
     */
    // TODO The parameter generalizeToStructure seems to be used only by some remaining kernel name generation
    // in LowerToLLVMPass. Once those call-sites have been refactored to use the kernel catalog, this feature
    // can be removed here.
    static std::string mlirTypeToCppTypeName(mlir::Type t, bool angleBrackets = true, bool generalizeToStructure = false) { // NOLINT(misc-no-recursion)
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
        else if(t.isa<mlir::daphne::StructureType>())
            return "Structure";
        else if(auto matTy = t.dyn_cast<mlir::daphne::MatrixType>()) {
            if(generalizeToStructure)
                return "Structure";
            else {
                switch (matTy.getRepresentation()) {
                    case mlir::daphne::MatrixRepresentation::Dense: {
                        const std::string vtName = mlirTypeToCppTypeName(matTy.getElementType(), angleBrackets, false);
                        return angleBrackets ? ("DenseMatrix<" + vtName + ">") : ("DenseMatrix_" + vtName);
                    }
                    case mlir::daphne::MatrixRepresentation::Sparse: {
                        const std::string vtName = mlirTypeToCppTypeName(matTy.getElementType(), angleBrackets, false);
                        return angleBrackets ? ("CSRMatrix<" + vtName + ">") : ("CSRMatrix_" + vtName);
                    }
                }
            }
        }
        else if(llvm::isa<mlir::daphne::FrameType>(t))
            if(generalizeToStructure)
                return "Structure";
            else
                return "Frame";
        else if(llvm::isa<mlir::daphne::StringType>(t))
            // This becomes "const char *" (which makes perfect sense for
            // strings) when inserted into the typical "const DT *" template of
            // kernel input parameters.
            return "char";
        else if(llvm::isa<mlir::daphne::DaphneContextType>(t))
            return "DaphneContext";
        else if(auto handleTy = t.dyn_cast<mlir::daphne::HandleType>()) {
            const std::string tName = mlirTypeToCppTypeName(handleTy.getDataType(), angleBrackets, generalizeToStructure);
            return angleBrackets ? ("Handle<" + tName + ">") : ("Handle_" + tName);
        }
        else if(llvm::isa<mlir::daphne::FileType>(t))
            return "File";
        else if(llvm::isa<mlir::daphne::DescriptorType>(t))
            return "Descriptor";
        else if(llvm::isa<mlir::daphne::TargetType>(t))
            return "Target";
        else if(auto memRefType = t.dyn_cast<mlir::MemRefType>()) {
            const std::string vtName = mlirTypeToCppTypeName(memRefType.getElementType(), angleBrackets, false);
            return angleBrackets ? ("StridedMemRefType<" + vtName + ",2>") : ("StridedMemRefType_" + vtName + "_2");
        }

        std::string typeName;
        llvm::raw_string_ostream rsos(typeName);
        t.print(rsos);
        throw std::runtime_error(
            "no C++ type name known for the given MLIR type: " + typeName
        );
    }

    static bool isMatrixComputation(mlir::Operation *v);

    /**
     * @brief Returns the DAPHNE context used in the given function.
     * 
     * Throws if there is not exactly one DAPHNE context.
     * 
     * @param func
     * @return 
     */
    [[maybe_unused]] mlir::Value static getDaphneContext(mlir::func::FuncOp & func) {
        mlir::Value dctx = nullptr;
        auto ops = func.getBody().front().getOps<mlir::daphne::CreateDaphneContextOp>();
        for(auto op : ops) {
            if(!dctx)
                dctx = op.getResult();
            else
                throw ErrorHandler::compilerError(op.getLoc(), "getDaphneContext",
                        "function body block contains more than one CreateDaphneContextOp"
                );
        }
        if(!dctx)
            throw ErrorHandler::compilerError(func.getLoc(), "getDaphneContext",
                    "function body block contains no CreateDaphneContextOp"
            );
        return dctx;
    }
    
    [[maybe_unused]] static bool isObjType(mlir::Type t) {
        return llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(t);
    }
    
    [[maybe_unused]] static bool hasObjType(mlir::Value v) {
        return isObjType(v.getType());
    }

    /**
     * @brief Returns the value type of the given scalar/matrix/frame type.
     * 
     * For matrices and frames, the value type is extracted. For scalars,
     * the type itself is the value type.
     * 
     * @param t the given scalar/matrix/frame type
     * @return the value type of the given type
     */
    static mlir::Type getValueType(mlir::Type t) {
        if(auto mt = t.dyn_cast<mlir::daphne::MatrixType>())
            return mt.getElementType();
        if(auto ft = t.dyn_cast<mlir::daphne::FrameType>())
            throw std::runtime_error("getValueType() doesn't support frames yet"); // TODO
        else // TODO Check if this is really a scalar.
            return t;
    }

    /**
     * @brief Sets the value type of the given scalar/matrix/frame type to the
     * given value type and returns this derived type.
     * 
     * For matrices and frames, the value type is set to the given value type.
     * For scalars, the given value type itself is returned.
     * 
     * @param t the scalar/matrix/frame type whose value type shall be set
     * @param vt the value type to use
     * @return the derived scalar/matrix/frame type
     */
    static mlir::Type setValueType(mlir::Type t, mlir::Type vt) {
        if(auto mt = t.dyn_cast<mlir::daphne::MatrixType>())
            return mt.withElementType(vt);
        if(auto ft = t.dyn_cast<mlir::daphne::FrameType>())
            throw std::runtime_error("setValueType() doesn't support frames yet"); // TODO
        else // TODO Check if this is really a scalar.
            return vt;
    }

    /**
     * @brief Checks if the two given types are the same, whereby
     * DaphneIR's unknown type acts as a wildcard.
     * 
     * The two types are considered equal, iff they are exactly the same
     * type, or one of the following "excuses" holds:
     * - at least one of the types is unknown
     * - both types are matrices and at least one of them has an unknown
     *   value type
     * 
     * @param t1 The first type
     * @param t2 The second type
     * @result `true` if the two types are considered equal, `false` otherwise
     */
    static bool equalUnknownAware(mlir::Type t1, mlir::Type t2) {
        using mlir::daphne::UnknownType;
        auto matT1 = t1.dyn_cast<mlir::daphne::MatrixType>();
        auto matT2 = t2.dyn_cast<mlir::daphne::MatrixType>();
        // The two given types are considered equal, iff:
        return (
            // The two types are exactly the same...
            t1 == t2
            // ...or one of the following "excuses" holds:
            || (
                // at least one of the types is unknown
                llvm::isa<UnknownType>(t1) || llvm::isa<UnknownType>(t2) ||
                // both types are matrices and at least one of them
                // has an unknown value type
                (matT1 && matT2 && (
                    llvm::isa<UnknownType>(matT1.getElementType()) ||
                    llvm::isa<UnknownType>(matT2.getElementType())
                ))
            )
        );
    }
};
