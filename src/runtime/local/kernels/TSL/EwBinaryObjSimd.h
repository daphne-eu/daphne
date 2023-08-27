/*
 * Copyright 2021 The DAPHNE Consortium
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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_TSL_CALC_H
#define SRC_RUNTIME_LOCAL_KERNELS_TSL_CALC_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>

#include <runtime/local/context/DaphneContext.h>
#include <util/memory/AlignmentHelper.hpp>

#include <util/constexpr/MemberDetector.h>

// #include "calc.h"

/// include the SIMD library
#include <tslintrin.hpp>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename TSL_ProcessingStyle, class DTRes, class DTLhs, class DTRhs>
struct EwBinaryObjSimd {
    static void apply(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx) = nullptr) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<typename TSL_ProcessingStyle, class DTRes, class DTLhs, class DTRhs>
void ewBinaryObjSimd(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx) = nullptr) {
    EwBinaryObjSimd<TSL_ProcessingStyle, DTRes, DTLhs, DTRhs>::apply(opCode, res, lhs, rhs, ctx);
}


// ****************************************************************************

template<typename TSL_ProcessingStyle, typename VT>
struct EwBinaryKernel {

    using register_type = typename TSL_ProcessingStyle::register_type;
    static constexpr size_t simd_element_count = TSL_ProcessingStyle::vector_element_count();
    static constexpr size_t simd_vector_size = TSL_ProcessingStyle::vector_size_B();

    template<template <typename ...> typename primitive, typename ... args_t>
    static register_type detected_and_execute(std::string name, args_t ... args){
        using primitiveSpezialization = primitive<TSL_ProcessingStyle, tsl::workaround>;

        if constexpr (! std::is_class_v<primitiveSpezialization>){
            throw std::runtime_error("Operation " + name + " not implemented for this SIMD type.");
        } else {
            if constexpr (daphne::detector::has_static_method_apply_v<primitiveSpezialization>){
                return primitiveSpezialization::apply(args...);
            }
            else {
                throw std::runtime_error("Operation " + name + " not implemented for this SIMD type.");
            }
        }
    }

    static register_type execute_primitive(BinaryOpCode opCode, register_type lhs, register_type rhs) {
        switch (opCode) {
            // Arithmetic
            case BinaryOpCode::ADD:
                return detected_and_execute<tsl::functors::add>("addition", lhs, rhs);
                // return tsl::add<TSL_ProcessingStyle>(lhs, rhs);
            case BinaryOpCode::SUB:
                return detected_and_execute<tsl::functors::sub>("subtraction", lhs, rhs);
                // return tsl::sub<TSL_ProcessingStyle>(lhs, rhs);
            case BinaryOpCode::MUL:
                return detected_and_execute<tsl::functors::mul>("multiplication", lhs, rhs);
                // return tsl::mul<TSL_ProcessingStyle>(lhs, rhs);
            case BinaryOpCode::DIV:
                return detected_and_execute<tsl::functors::div>("division", lhs, rhs);
            case BinaryOpCode::MOD:
                return detected_and_execute<tsl::functors::mod>("modulo", lhs, rhs);
            case BinaryOpCode::POW:
                return detected_and_execute<tsl::functors::pow>("power", lhs, rhs);

            // Comparison
            case BinaryOpCode::EQ:
                return detected_and_execute<tsl::functors::equal>("equal", lhs, rhs);
            case BinaryOpCode::NEQ:
                return detected_and_execute<tsl::functors::nequal>("not equal", lhs, rhs);
            case BinaryOpCode::LT:  
                return detected_and_execute<tsl::functors::less_than>("less than", lhs, rhs);
            case BinaryOpCode::LE:
                return detected_and_execute<tsl::functors::less_than_or_equal>("less equal", lhs, rhs);
            case BinaryOpCode::GT:
                return detected_and_execute<tsl::functors::greater_than>("greater than", lhs, rhs);
            case BinaryOpCode::GE:
                return detected_and_execute<tsl::functors::greater_than_or_equal>("greater equal", lhs, rhs);
                
            // Min/Max
            case BinaryOpCode::MIN:
                return detected_and_execute<tsl::functors::min>("minimum", lhs, rhs);
            case BinaryOpCode::MAX:
                return detected_and_execute<tsl::functors::max>("maximum", lhs, rhs);

            // Logical
            case BinaryOpCode::AND:
                // return detected_and_execute<tsl::functors::and_>("logical and", lhs, rhs);
            case BinaryOpCode::OR:
                // return detected_and_execute<tsl::functors::or_>("logical or", lhs, rhs);

            default:
                throw std::runtime_error("Unknown binary operation or not implemented, yet.");
        }
    }

    static void apply(BinaryOpCode opCode, VT * res, const VT * lhs, const VT * rhs, uint64_t iterations){
        register_type reg_lhs, reg_rhs, reg_res;
        
        const AlignmentHelper::Alignment alignment_lhs = AlignmentHelper::getAlignment(lhs, simd_vector_size);
        const AlignmentHelper::Alignment alignment_rhs = AlignmentHelper::getAlignment(rhs, simd_vector_size);
        const AlignmentHelper::Alignment alignment_res = AlignmentHelper::getAlignment(res, simd_vector_size);

        for (uint64_t i = 0; i < iterations; i++) {
            /// load data into SIMD register
            reg_lhs = alignment_lhs.isAligned() 
                ? tsl::load <TSL_ProcessingStyle>(lhs + i * TSL_ProcessingStyle::vector_element_count()) 
                : tsl::loadu<TSL_ProcessingStyle>(lhs + i * TSL_ProcessingStyle::vector_element_count());
            reg_rhs = alignment_rhs.isAligned()
                ? tsl::load <TSL_ProcessingStyle>(rhs + i * TSL_ProcessingStyle::vector_element_count())
                : tsl::loadu<TSL_ProcessingStyle>(rhs + i * TSL_ProcessingStyle::vector_element_count());

            /// execute primitive
            reg_res = execute_primitive(opCode, reg_lhs, reg_rhs);

            /// store result
            alignment_res.isAligned()
                ? tsl::store <TSL_ProcessingStyle>(res + i * TSL_ProcessingStyle::vector_element_count(), reg_res)
                : tsl::storeu<TSL_ProcessingStyle>(res + i * TSL_ProcessingStyle::vector_element_count(), reg_res);
        }
    }

};


// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename TSL_ProcessingStyle, typename VT>
struct EwBinaryObjSimd<TSL_ProcessingStyle, DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {

    /// shortcuts for the processing style
    using simd_register_type = typename TSL_ProcessingStyle::register_type;
    static constexpr size_t simd_element_count = TSL_ProcessingStyle::vector_element_count();
    static constexpr size_t simd_vector_size = TSL_ProcessingStyle::vector_size_B();
    
    /// alias for scalar processing style
    using TSL_ScalarPS = typename tsl::simd<typename TSL_ProcessingStyle::base_type , tsl::scalar>;
    
    /**
     * @brief Applies the given binary operation to the given matrices and stores the result in the given result matrix.
     * 
     * This function inteprets the given matrices as vectors and applies the given binary operation to the corresponding elements.
     * 
     * @param opCode 
     * @param res 
     * @param lhs 
     * @param rhs 
     */
    static void apply(BinaryOpCode opCode, DenseMatrix<VT> *& res, const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx) = nullptr) {
        /// assert inputs are of the same size
        assert(lhs->getNumRows() == rhs->getNumRows());
        assert(lhs->getNumCols() == rhs->getNumCols());

        const size_t rows = lhs->getNumRows();
        const size_t cols = lhs->getNumCols();
        size_t size = rows * cols;
        /// @TODO Skip handling -> rowSkip > colNum

        /// allocate the result matrix if necessary
        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(rows, cols, false);
        }

        /// get the data pointers
        const VT * lhsData = lhs->getValues();
        const VT * rhsData = rhs->getValues();
        VT * resData = res->getValues();


        /// process until the data is aligned, if possible
        auto lhs_alignment = AlignmentHelper::getAlignment(lhsData, simd_vector_size);
        auto rhs_alignment = AlignmentHelper::getAlignment(rhsData, simd_vector_size);


        /// Scalar processing unaligned data 
        uint64_t unaligned_iterations = std::min(std::min(lhs_alignment.getOffset(), rhs_alignment.getOffset()), size);
        EwBinaryKernel<TSL_ScalarPS, VT>::apply(opCode, resData, lhsData, rhsData, unaligned_iterations);
        size    -= unaligned_iterations;
        resData += unaligned_iterations; 
        lhsData += unaligned_iterations; 
        rhsData += unaligned_iterations;

        
        /// update alignment
        lhs_alignment = AlignmentHelper::getAlignment(lhsData, simd_vector_size);
        rhs_alignment = AlignmentHelper::getAlignment(rhsData, simd_vector_size);


        /// SIMD processing
        uint64_t iterations = size / simd_element_count;
        uint64_t remainder  = size % simd_element_count;

        EwBinaryKernel<TSL_ProcessingStyle, VT>::apply(opCode, resData, lhsData, rhsData, iterations);
        resData += iterations * simd_element_count;
        lhsData += iterations * simd_element_count;
        rhsData += iterations * simd_element_count;

        /// Scalar remainder processing
        EwBinaryKernel<TSL_ScalarPS, VT>::apply(opCode, resData, lhsData, rhsData, remainder);

    }
};





#endif //SRC_RUNTIME_LOCAL_KERNELS_TSL_CALC_H
