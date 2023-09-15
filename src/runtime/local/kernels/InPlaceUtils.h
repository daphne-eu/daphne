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


#ifndef SRC_RUNTIME_LOCAL_UTILS_RUNTIMEUTILS_H
#define SRC_RUNTIME_LOCAL_UTILS_RUNTIMEUTILS_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <ir/daphneir/Daphne.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <optional>

#include <stdexcept>
#include <string>
#include <iostream>
#include <type_traits>

class InPlaceUtils {
public:

    /**
    * @brief Checks if two matrices are of the same type and have the same dimensions.
    *
    * @param arg1 Pointer to a matrix.
    * @param arg2 Pointer to a matrix.
    * @return Returns true if the matrices are of the same type and have the same dimensions.
    */
    template<typename VTLhs, typename VTRhs>
    static bool isValidType(const DenseMatrix<VTLhs>* arg1, const DenseMatrix<VTRhs>* arg2) {
        if(arg1->getNumCols() == arg2->getNumCols() && arg1->getNumRows() == arg2->getNumRows()) {
            return std::is_same_v<VTLhs, VTRhs>; 
        }
        return false;
    }

    /**
    * @brief Checks if two matrices are of the same type and have atleast complementary dimensions.
    *
    * @param arg1 Pointer to a matrix.
    * @param arg2 Pointer to a matrix.
    * @return Returns true if the matrices are of the same type and complementary dimensions. 
    */
    template<typename VTLhs, typename VTRhs>
    static bool isValidTypeWeak(const DenseMatrix<VTLhs>* arg1, const DenseMatrix<VTRhs>* arg2) {
        //if (arg1->getNumCols() * arg1->getNumRows() == arg2->getNumRows() * arg2->getNumCols()) {
        if (arg1->getNumItems() == arg2->getNumItems()) {
            return std::is_same_v<VTLhs, VTRhs>;
        }
        return false;
    }

    /**
    * @brief From one or multiple operands, the function decides whether it can and which operand is used as the result pointer.
    *
    * @param arg Pointer to a matrix.
    * @param hasFutureUseArg Denotes if the compiler has determined that the result of the operation is used in a future operation.
    * @param args Additional matrices and hasFutureUse
    * @return Returns the first operand that can be used as the result pointer.
    */
    template<typename VTArg, typename... Args>
    static DenseMatrix<VTArg>* selectInPlaceableOperand(DenseMatrix<VTArg> *arg, bool hasFutureUseArg, Args... args) {
        if (InPlaceUtils::isInPlaceable(arg, hasFutureUseArg))
            return arg;

        if constexpr (sizeof...(Args) == 0) {
            return nullptr;
        } else {
            return selectInPlaceableOperand(args...);
        }
    }

    /**
    * @brief Checks if a matrix can be used as the result pointer. 
    *        TODO: As currently the MetaDataObject needs stabilisation and 
    *              CSRMatrix and DenseMatrices are storing the informatin in a different way, 
    *              we only look at the values (DenseMatrix) use count of the underlying shared_ptr. 
    *
    * @param arg Pointer to a matrix.
    * @param hasFutureUseArg Denotes if the compiler has determined that the result of the operation is used in a future operation.
    * @return Returns true if the matrix can be used as the result pointer.
    */
    template<typename VTArg>
    static bool isInPlaceable(DenseMatrix<VTArg> *arg, bool hasFutureUseArg) {
        if (!hasFutureUseArg) {
            if (arg->getRefCounter() == 1 && arg->getValuesUseCount() == 1) {
                return true;
            }
        }
        return false;
    }
};

#endif //SRC_RUNTIME_LOCAL_UTILS_RUNTIMEUTILS_H