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

#include <compiler/inference/TypeInferenceUtils.h>

#include <llvm/Support/raw_os_ostream.h>

#include <tags.h>

// TODO Make this a general (test) utility.
/**
 * Allows catch2 can print the name of MLIR types in its error messages.
 * @param os
 * @param type
 * @return 
 */
std::ostream & operator<<(std::ostream & os, mlir::Type const & type) {
    llvm::raw_os_ostream raw(os);
    const_cast<mlir::Type &>(type).print(raw);
    return os;
}

#include <catch.hpp>

#include <type_traits>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utlities
// ****************************************************************************

// TODO Move this somewhere central?
/**
 * @brief Utility to find out if two C++ types are instantiations of the same
 * template.
 */
template<class, class>
struct is_same_template : std::false_type {};
template<template<class> class Trait, class ConcreteOp1, class ConcreteOp2>
struct is_same_template<Trait<ConcreteOp1>, Trait<ConcreteOp2>> : std::true_type {};

/**
 * @brief Mocks an `mlir::Value`, such that we can use its interface in unit
 * tests without fully wiring a real value.
 */
class ValueMock {
    Type type;
    
public:
    ValueMock(Type type) : type(type) {
        //
    }
    
    Type getType() const {
        return type;
    }
};

// TODO Instead of a single data type trait and value type trait, we could use
// variadic templates to attach an arbitrary number of traits. This could make
// this class more useful in other test cases.
/**
 * @brief Mocks an `mlir::Operation`, such that we can use its interface in
 * unit tests without fully wiring a real operation.
 */
template<template<class> class DataTypeTrait, template<class> class ValueTypeTrait>
class OperationMock {
    MLIRContext * context;
    std::vector<Type> operandTypes;
    
public:
    OperationMock(MLIRContext * context, std::vector<Type> operandTypes)
    : context(context), operandTypes(operandTypes) {
        //
    }
    
    MLIRContext * getContext() const {
        return context;
    }
    
    template<template<class ConcreteOp> class Trait>
    bool hasTrait() const {
        // TODO Instantiating the traits for int could break...
        return is_same_template<Trait<int>, DataTypeTrait<int>>::value
                || is_same_template<Trait<int>, ValueTypeTrait<int>>::value;
    }
    
    std::vector<Type> getOperandTypes() const {
        return operandTypes;
    }
    
    ValueMock getOperand(unsigned idx) const {
        return ValueMock(operandTypes[idx]);
    }
};

// ****************************************************************************
// Macros simplifying test case definitions
// ****************************************************************************

#define MAKE_CASE(DataTypeTrait, ValueTypeTrait, operandTypes, expectedType) \
    { \
        OperationMock<DataTypeTrait, ValueTypeTrait> op(&ctx, operandTypes); \
        CHECK(inferTypeByTraits(&op) == expectedType); \
    }

#define MAKE_CASE_COMMUTATIVE(DataTypeTrait, ValueTypeTrait, operandType0, operandType1, expectedType) \
    MAKE_CASE(DataTypeTrait, ValueTypeTrait, ONE({operandType0, operandType1}), expectedType) \
    MAKE_CASE(DataTypeTrait, ValueTypeTrait, ONE({operandType1, operandType0}), expectedType)

#define MAKE_CASE_THROWS(DataTypeTrait, ValueTypeTrait, operandTypes) \
    { \
        OperationMock<DataTypeTrait, ValueTypeTrait> op(&ctx, operandTypes); \
        CHECK_THROWS(inferTypeByTraits(&op)); \
    }

// TODO Make this a general util.
#define ONE(...) __VA_ARGS__

// ****************************************************************************
// Test cases
// ****************************************************************************

TEST_CASE("TypeInferenceTraits", TAG_INFERENCE) {
    MLIRContext ctx;
    ctx.getOrLoadDialect<daphne::DaphneDialect>();
    
    OpBuilder builder(&ctx);
    
    // TODO Can we have variadic args, such that we don't need {} at call-sites?
    // Utility function for creating a frame type with the specified column
    // types.
    auto frm = [&ctx](std::vector<Type> columnTypes) {
        return daphne::FrameType::get(&ctx, columnTypes);
    };
    
    // A few scalar types for convenient use below.
    [[maybe_unused]] Type u = daphne::UnknownType::get(&ctx);
    [[maybe_unused]] Type str = daphne::StringType::get(&ctx);
    [[maybe_unused]] Type f64 = builder.getF64Type();
    [[maybe_unused]] Type f32 = builder.getF32Type();
    [[maybe_unused]] Type ui64 = builder.getIntegerType(64, false);
    [[maybe_unused]] Type ui32 = builder.getIntegerType(32, false);
    [[maybe_unused]] Type ui8 = builder.getIntegerType(8, false);
    [[maybe_unused]] Type si64 = builder.getIntegerType(64, true);
    [[maybe_unused]] Type si32 = builder.getIntegerType(32, true);
    [[maybe_unused]] Type si8 = builder.getIntegerType(8, true);
    [[maybe_unused]] Type bl = builder.getI1Type();
    
    // A few matrix types for convenient use below.
    [[maybe_unused]] daphne::MatrixType matu = daphne::MatrixType::get(&ctx, u);
    [[maybe_unused]] daphne::MatrixType matstr = daphne::MatrixType::get(&ctx, str);
    [[maybe_unused]] daphne::MatrixType matf64 = daphne::MatrixType::get(&ctx, f64);
    [[maybe_unused]] daphne::MatrixType matf32 = daphne::MatrixType::get(&ctx, f32);
    [[maybe_unused]] daphne::MatrixType matui64 = daphne::MatrixType::get(&ctx, ui64);
    [[maybe_unused]] daphne::MatrixType matui32 = daphne::MatrixType::get(&ctx, ui32);
    [[maybe_unused]] daphne::MatrixType matui8 = daphne::MatrixType::get(&ctx, ui8);
    [[maybe_unused]] daphne::MatrixType matsi64 = daphne::MatrixType::get(&ctx, si64);
    [[maybe_unused]] daphne::MatrixType matsi32 = daphne::MatrixType::get(&ctx, si32);
    [[maybe_unused]] daphne::MatrixType matsi8 = daphne::MatrixType::get(&ctx, si8);
    [[maybe_unused]] daphne::MatrixType matbl = daphne::MatrixType::get(&ctx, bl);
    
    // A few frame types for convenient use below.
    [[maybe_unused]] Type frmf64 = frm({f64});
    
    //-------------------------------------------------------------------------
    // Tests for various combinations of data/value type traits
    //-------------------------------------------------------------------------
    
    // TypeFromFirstArg is actually just one trait, but for the test setup we
    // assign it twice (which doesn't hurt).
    #define DTT TypeFromFirstArg
    #define VTT TypeFromFirstArg
    // exactly one arg -> should be retained
    MAKE_CASE(DTT, VTT, {f64}, f64)
    MAKE_CASE(DTT, VTT, {matf64}, matf64)
    MAKE_CASE(DTT, VTT, {frmf64}, frmf64)
    MAKE_CASE(DTT, VTT, {u}, u)
    // more than one arg -> additional args shouldn't interfere
    MAKE_CASE(DTT, VTT, ONE({f64, matf64}), f64)
    MAKE_CASE(DTT, VTT, ONE({matf64, frmf64}), matf64)
    MAKE_CASE(DTT, VTT, ONE({frmf64, u}), frmf64)
    MAKE_CASE(DTT, VTT, ONE({u, f64}), u)
    #undef DTT
    #undef VTT

    #define DTT DataTypeFromFirstArg
    #define VTT ValueTypeFromFirstArg
    // exactly one arg -> should be retained
    MAKE_CASE(DTT, VTT, {f64}, f64)
    MAKE_CASE(DTT, VTT, {matf64}, matf64)
    MAKE_CASE(DTT, VTT, {frmf64}, frmf64)
    MAKE_CASE(DTT, VTT, {u}, u)
    // more than one arg -> additional args shouldn't interfere
    MAKE_CASE(DTT, VTT, ONE({f64, matf64}), f64)
    MAKE_CASE(DTT, VTT, ONE({matf64, frmf64}), matf64)
    MAKE_CASE(DTT, VTT, ONE({frmf64, u}), frmf64)
    MAKE_CASE(DTT, VTT, ONE({u, f64}), u)
    #undef DTT
    #undef VTT
    
    #define DTT DataTypeFromArgs
    #define VTT ValueTypeFromArgs
    // one arg -> should be retained
    MAKE_CASE(DTT, VTT, {f64}, f64)
    MAKE_CASE(DTT, VTT, {matf64}, matf64)
    MAKE_CASE(DTT, VTT, {frmf64}, frmf64)
    MAKE_CASE(DTT, VTT, {u}, u)
    // two args, mixed data types, same value type
    MAKE_CASE_COMMUTATIVE(DTT, VTT, f64, matf64, matf64)
    MAKE_CASE_COMMUTATIVE(DTT, VTT, f64, frmf64, frmf64)
    MAKE_CASE_COMMUTATIVE(DTT, VTT, f64, u, u)
    MAKE_CASE_COMMUTATIVE(DTT, VTT, matf64, frmf64, frmf64)
    MAKE_CASE_COMMUTATIVE(DTT, VTT, matf64, u, u)
    MAKE_CASE_COMMUTATIVE(DTT, VTT, frmf64, u, u)
    // two args, same data type, mixed value types
        // scalar
            // unsigned integer bit width
            MAKE_CASE_COMMUTATIVE(DTT, VTT, bl, ui8, ui8)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, ui8, ui32, ui32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, ui32, ui64, ui64)
            // signed integer bit width
            MAKE_CASE_COMMUTATIVE(DTT, VTT, bl, si8, si8)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si8, si32, si32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si32, si64, si64)
            // signed vs unsigned integer
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si8, ui8, ui8)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si32, ui32, ui32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si64, ui64, ui64)
            // floating-point precision
            MAKE_CASE_COMMUTATIVE(DTT, VTT, f32, f64, f64)
            // floating point vs integer
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si32, f32, f32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si64, f32, f32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si64, f64, f64)
            // string vs other types
            MAKE_CASE_COMMUTATIVE(DTT, VTT, si64, str, str)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, ui64, str, str)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, f64, str, str)
            // unknown vs other types
            MAKE_CASE_COMMUTATIVE(DTT, VTT, u, ui64, u)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, u, si64, u)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, u, f64, u)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, u, str, u)
        // matrix (same cases as for scalars, just with matrices)
            // unsigned integer bit width
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matbl, matui8, matui8)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matui8, matui32, matui32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matui32, matui64, matui64)
            // signed integer bit width
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matbl, matsi8, matsi8)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi8, matsi32, matsi32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi32, matsi64, matsi64)
            // signed vs unsigned integer
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi8, matui8, matui8)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi32, matui32, matui32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi64, matui64, matui64)
            // floating-point precision
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matf32, matf64, matf64)
            // floating point vs integer
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi32, matf32, matf32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi64, matf32, matf32)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi64, matf64, matf64)
            // string vs other types
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matsi64, matstr, matstr)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matui64, matstr, matstr)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matf64, matstr, matstr)
            // unknown vs other types
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matu, matui64, matu)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matu, matsi64, matu)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matu, matf64, matu)
            MAKE_CASE_COMMUTATIVE(DTT, VTT, matu, matstr, matu)
        // frame
        MAKE_CASE_COMMUTATIVE(
                DTT, VTT,
                frm({ui8, f64, u, si64}), frm({si32, f32, f64, str}),
                frm({si32, f64, u, str})
        )
    // multiple args, mixed data types, mixed value types
    MAKE_CASE(DTT, VTT, ONE({si64, matf32, frmf64, u}), u)
    MAKE_CASE(DTT, VTT, ONE({matf32, f64, si32}), matf64)
    MAKE_CASE(DTT, VTT, ONE({si32, matu, str}), matu)
    MAKE_CASE(DTT, VTT, ONE({matf32, frm({ui8, u, str}), f64}), frm({f64, u, str}))
    MAKE_CASE(DTT, VTT, ONE({str, matu, frm({f64, str})}), frm({u, u}))
    // frames with different number of columns -> not allowed
    MAKE_CASE_THROWS(DTT, VTT, ONE({frmf64, frm({f32, si64})}))
    MAKE_CASE_THROWS(DTT, VTT, ONE({frmf64, frm({f32, si64}), f64}))
    MAKE_CASE_THROWS(DTT, VTT, ONE({frm({f32, si64}), frm({ui8, ui32, ui64})}))
    #undef DTT
    #undef VTT
    
    #define DTT DataTypeFromFirstArg
    #define VTT ValueTypeFromArgsFP
    // one argument
        // scalar
            // bool becomes most general floating-point
            MAKE_CASE(DTT, VTT, {bl}, f64)
            // integers become most general floating-point
            MAKE_CASE(DTT, VTT, {si8}, f64)
            MAKE_CASE(DTT, VTT, {si32}, f64)
            MAKE_CASE(DTT, VTT, {si64}, f64)
            MAKE_CASE(DTT, VTT, {ui8}, f64)
            MAKE_CASE(DTT, VTT, {ui32}, f64)
            MAKE_CASE(DTT, VTT, {ui64}, f64)
            // floating-point stays as-is
            MAKE_CASE(DTT, VTT, {f32}, f32)
            MAKE_CASE(DTT, VTT, {f64}, f64)
            // string becomes most general floating-point
            MAKE_CASE(DTT, VTT, {str}, f64)
            // unknown stays unknown
            MAKE_CASE(DTT, VTT, {u}, u)
        // matrix
            // bool becomes most general floating-point
            MAKE_CASE(DTT, VTT, {matbl}, matf64)
            // integers become most general floating-point
            MAKE_CASE(DTT, VTT, {matsi8}, matf64)
            MAKE_CASE(DTT, VTT, {matsi32}, matf64)
            MAKE_CASE(DTT, VTT, {matsi64}, matf64)
            MAKE_CASE(DTT, VTT, {matui8}, matf64)
            MAKE_CASE(DTT, VTT, {matui32}, matf64)
            MAKE_CASE(DTT, VTT, {matui64}, matf64)
            // floating-point stays as-is
            MAKE_CASE(DTT, VTT, {matf32}, matf32)
            MAKE_CASE(DTT, VTT, {matf64}, matf64)
            // string becomes most general floating-point
            MAKE_CASE(DTT, VTT, {matstr}, matf64)
            // unknown stays unknown
            MAKE_CASE(DTT, VTT, {matu}, matu)
        // frame
        MAKE_CASE(
                DTT, VTT,
                {frm({bl, ui8, si32, f32, f64, str, u})},
                frm({f64, f64, f64, f32, f64, f64, u})
        )
    // multiple args
        // scalar
        MAKE_CASE(DTT, VTT, ONE({ui8, si32}), f64)
        MAKE_CASE(DTT, VTT, ONE({f32, f64}), f64)
        MAKE_CASE(DTT, VTT, ONE({si32, f32}), f32)
        MAKE_CASE(DTT, VTT, ONE({si32, f32, str}), f64)
        MAKE_CASE(DTT, VTT, ONE({si32, f32, u}), u)
        // matrix
        MAKE_CASE(DTT, VTT, ONE({matui8, matsi32}), matf64)
        MAKE_CASE(DTT, VTT, ONE({matf32, matf64}), matf64)
        MAKE_CASE(DTT, VTT, ONE({matsi32, matf32}), matf32)
        MAKE_CASE(DTT, VTT, ONE({matsi32, matf32, matstr}), matf64)
        MAKE_CASE(DTT, VTT, ONE({matsi32, matf32, matu}), matu)
        // frame
        MAKE_CASE(
                DTT, VTT,
                ONE({frm({ui8, f32, str}), frm({si32, si8, u})}),
                frm({f64, f32, u})
        )
        // mixed
        MAKE_CASE(DTT, VTT, ONE({matf32, si32}), matf32)
        MAKE_CASE(DTT, VTT, ONE({matsi32, si64}), matf64)
        MAKE_CASE(DTT, VTT, ONE({frm({ui8, f64}), matf32}), frm({f32, f64}))
    #undef DTT
    #undef VTT
    
    #define DTT DataTypeFromFirstArg
    #define VTT ValueTypeFromArgsInt
    // one argument
        // scalar
            // bool stays bool
            MAKE_CASE(DTT, VTT, {bl}, bl) // TODO Is this what we want?
            // integers stay as-is
            MAKE_CASE(DTT, VTT, {si8}, si8)
            MAKE_CASE(DTT, VTT, {si32}, si32)
            MAKE_CASE(DTT, VTT, {si64}, si64)
            MAKE_CASE(DTT, VTT, {ui8}, ui8)
            MAKE_CASE(DTT, VTT, {ui32}, ui32)
            MAKE_CASE(DTT, VTT, {ui64}, ui64)
            // floating-point becomes most general integer
            MAKE_CASE(DTT, VTT, {f32}, ui64)
            MAKE_CASE(DTT, VTT, {f64}, ui64)
            // string becomes most general integer
            MAKE_CASE(DTT, VTT, {str}, ui64)
            // unknown stays unknown
            MAKE_CASE(DTT, VTT, {u}, u)
        // matrix
            // bool stays bool
            MAKE_CASE(DTT, VTT, {matbl}, matbl) // TODO Is this what we want?
            // integers stay as-is
            MAKE_CASE(DTT, VTT, {matsi8}, matsi8)
            MAKE_CASE(DTT, VTT, {matsi32}, matsi32)
            MAKE_CASE(DTT, VTT, {matsi64}, matsi64)
            MAKE_CASE(DTT, VTT, {matui8}, matui8)
            MAKE_CASE(DTT, VTT, {matui32}, matui32)
            MAKE_CASE(DTT, VTT, {matui64}, matui64)
            // floating-point becomes most general integer
            MAKE_CASE(DTT, VTT, {matf32}, matui64)
            MAKE_CASE(DTT, VTT, {matf64}, matui64)
            // string becomes most general integer
            MAKE_CASE(DTT, VTT, {matstr}, matui64)
            // unknown stays unknown
            MAKE_CASE(DTT, VTT, {matu}, matu)
        // frame
        MAKE_CASE(
                DTT, VTT,
                {frm({bl, ui8, si32, f32, f64, str, u})},
                frm({bl, ui8, si32, ui64, ui64, ui64, u})
        ) // TODO Is this what we want regarding bool?
    // multiple args
        // scalar
        MAKE_CASE(DTT, VTT, ONE({ui8, si32}), si32)
        MAKE_CASE(DTT, VTT, ONE({f32, f64}), ui64)
        MAKE_CASE(DTT, VTT, ONE({si32, f32}), ui64)
        MAKE_CASE(DTT, VTT, ONE({si32, f32, str}), ui64)
        MAKE_CASE(DTT, VTT, ONE({si32, f32, u}), u)
        // matrix
        MAKE_CASE(DTT, VTT, ONE({matui8, matsi32}), matsi32)
        MAKE_CASE(DTT, VTT, ONE({matf32, matf64}), matui64)
        MAKE_CASE(DTT, VTT, ONE({matsi32, matf32}), matui64)
        MAKE_CASE(DTT, VTT, ONE({matsi32, matf32, matstr}), matui64)
        MAKE_CASE(DTT, VTT, ONE({matsi32, matf32, matu}), matu)
        // frame
        MAKE_CASE(
                DTT, VTT,
                ONE({frm({ui8, f32, str}), frm({si32, si8, u})}),
                frm({si32, ui64, u})
        )
        // mixed
        MAKE_CASE(DTT, VTT, ONE({matf32, si32}), matui64)
        MAKE_CASE(DTT, VTT, ONE({matsi32, si64}), matsi64)
        MAKE_CASE(DTT, VTT, ONE({frm({ui8, f64}), matf32}), frm({ui64, ui64}))
    #undef DTT
    #undef VTT
    
    #define DTT DataTypeFromArgs
    #define VTT ValueTypesConcat
    // one arg -> not allowed
    MAKE_CASE_THROWS(DTT, VTT, {f64})
    MAKE_CASE_THROWS(DTT, VTT, {matf64})
    MAKE_CASE_THROWS(DTT, VTT, {frmf64})
    MAKE_CASE_THROWS(DTT, VTT, {u})
    // two args
        // same data type, mixed value types
        MAKE_CASE(DTT, VTT, ONE({f32, f64}), f64)
        MAKE_CASE(DTT, VTT, ONE({f32, u}), u)
        MAKE_CASE(DTT, VTT, ONE({matf32, matf64}), matf64)
        MAKE_CASE(DTT, VTT, ONE({matf32, matu}), matu)
        MAKE_CASE(DTT, VTT, ONE({frm({f32}), frm({f64})}), frm({f32, f64}))
        MAKE_CASE(
                DTT, VTT,
                ONE({frm({f32, u, si64}), frm({f64, f32})}),
                frm({f32, u, si64, f64, f32})
        )
        // same mixed type, mixed value types
        MAKE_CASE(DTT, VTT, ONE({matf32, f64}), matf64)
        MAKE_CASE(DTT, VTT, ONE({f32, frm({f64, si64})}), frm({f32, f64, si64}))
        // TODO How to properly represent this case (see #421)?
//        MAKE_CASE(
//                DTT, VTT,
//                // note: matrix without shape information
//                ONE({matf32, frm({f64, si64})}),
//                frm({u})
//        )
        MAKE_CASE(DTT, VTT, ONE({matf32.withShape(-1, 3), frm({f64, si64})}), frm({f32, f32, f32, f64, si64}))
        MAKE_CASE(DTT, VTT, ONE({frm({f64, si64}), matf32.withShape(-1, 3)}), frm({f64, si64, f32, f32, f32}))
    // more than two args -> additional args shouldn't impact the value type
    // (maybe we change this later) (same cases as above, just more args)
        // same data type, mixed value types
        MAKE_CASE(DTT, VTT, ONE({f32, f64, str}), f64)
        MAKE_CASE(DTT, VTT, ONE({f32, u, str}), u)
        MAKE_CASE(DTT, VTT, ONE({matf32, matf64, matstr}), matf64)
        MAKE_CASE(DTT, VTT, ONE({matf32, matu, matstr}), matu)
        MAKE_CASE(DTT, VTT, ONE({frm({f32}), frm({f64}), frm({str})}), frm({f32, f64}))
        MAKE_CASE(
                DTT, VTT,
                ONE({frm({f32, u, si64}), frm({f64, f32}), frm({str})}),
                frm({f32, u, si64, f64, f32})
        )
        // same mixed type, mixed value types
        MAKE_CASE(DTT, VTT, ONE({matf32, f64, matstr}), matf64)
        MAKE_CASE(
                DTT, VTT,
                ONE({f32, frm({f64, si64}), frm({str})}),
                frm({f32, f64, si64})
        )
        // TODO How to properly represent this case (see #421)?
//        MAKE_CASE(
//                DTT, VTT,
//                // note: matrix without shape information
//                ONE({matf32, frm({f64, si64}), frm({str})}),
//                frm({u})
//        )
        MAKE_CASE(
                DTT, VTT,
                ONE({matf32.withShape(-1, 3), frm({f64, si64}), str}),
                frm({f32, f32, f32, f64, si64})
        )
        MAKE_CASE(
                DTT, VTT,
                ONE({frm({f64, si64}), matf32.withShape(-1, 3), str}),
                frm({f64, si64, f32, f32, f32})
        )
    #undef DTT
    #undef VTT

    #define DTT DataTypeSca
    #define VTT ValueTypeFromFirstArg
        // Mainly test if result is scalar and input frame column types are
        // collapsed correctly.
        MAKE_CASE(DTT, VTT, {f64}, f64)
        MAKE_CASE(DTT, VTT, {matf64}, f64)
        MAKE_CASE(DTT, VTT, {frmf64}, f64)
        MAKE_CASE(DTT, VTT, {frm({f64, f32, str})}, str)
        MAKE_CASE(DTT, VTT, {u}, u)
    #undef DTT
    #undef VTT

    #define DTT DataTypeMat
    #define VTT ValueTypeFromFirstArg
        // Mainly test if result is matrix and input frame column types are
        // collapsed correctly.
        MAKE_CASE(DTT, VTT, {f64}, matf64)
        MAKE_CASE(DTT, VTT, {matf64}, matf64)
        MAKE_CASE(DTT, VTT, {frmf64}, matf64)
        MAKE_CASE(DTT, VTT, {frm({f64, f32, str})}, matstr)
        MAKE_CASE(DTT, VTT, {u}, matu)
    #undef DTT
    #undef VTT

    #define DTT DataTypeFrm
    #define VTT ValueTypeFromFirstArg
        // Mainly test if result is frame and result column types are derived
        // correctly.
        MAKE_CASE(DTT, VTT, {f64}, frmf64)
        // TODO How to properly represent this case (see #421)?
//        MAKE_CASE(DTT, VTT, {matf64}, frm({u})) // note: matrix without shape information
        MAKE_CASE(DTT, VTT, {matf64.withShape(-1, 3)}, frm({f64, f64, f64}))
        MAKE_CASE(DTT, VTT, {frmf64}, frmf64)
        MAKE_CASE(DTT, VTT, {frm({f64, f32, str})}, frm({f64, f32, str}))
        // TODO How to properly represent this case (see #421)?
//        MAKE_CASE(DTT, VTT, {u}, frm({u}))
    #undef DTT
    #undef VTT
}
