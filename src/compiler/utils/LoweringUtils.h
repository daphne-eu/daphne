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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#pragma GCC diagnostic pop
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

mlir::Value insertMemRefAlloc(mlir::MemRefType type, mlir::Location loc,
                              mlir::PatternRewriter &rewriter);

void insertMemRefDealloc(mlir::Value memref, mlir::Location loc,
                         mlir::PatternRewriter &rewriter);

void affineFillMemRefInt(int value, mlir::ConversionPatternRewriter &rewriter,
                         mlir::Location loc, mlir::ArrayRef<int64_t> shape,
                         mlir::MLIRContext *ctx, mlir::Value memRef,
                         mlir::Type elemType);

void affineFillMemRefInt(mlir::Value value,
                         mlir::ConversionPatternRewriter &rewriter,
                         mlir::Location loc, mlir::ArrayRef<int64_t> shape,
                         mlir::MLIRContext *ctx, mlir::Value memRef);

void affineFillMemRef(double value, mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, mlir::ArrayRef<int64_t> shape,
                      mlir::MLIRContext *ctx, mlir::Value memRef,
                      mlir::Type elemType);

mlir::Value convertMemRefToDenseMatrix(mlir::Location,
                                       mlir::ConversionPatternRewriter &,
                                       mlir::Value memRef, mlir::Type);

llvm::Optional<mlir::Value> materializeCastFromIllegal(mlir::OpBuilder &builder,
                                                       mlir::Type type,
                                                       mlir::ValueRange inputs,
                                                       mlir::Location loc);

llvm::Optional<mlir::Value> materializeCastToIllegal(mlir::OpBuilder &builder,
                                                     mlir::Type type,
                                                     mlir::ValueRange inputs,
                                                     mlir::Location loc);

mlir::Type convertFloat(mlir::FloatType floatType);

mlir::Type convertInteger(mlir::IntegerType intType);

mlir::Operation *findLastUseOfSSAValue(mlir::Value &v);
