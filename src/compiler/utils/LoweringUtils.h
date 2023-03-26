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

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"

mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc,
                                  mlir::PatternRewriter &rewriter);

void affineFillMemRefInt(int value, mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, mlir::ArrayRef<int64_t> shape,
                      mlir::MLIRContext *ctx, mlir::Value memRef,
                      mlir::Type elemType);

void affineFillMemRef(double value, mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, mlir::ArrayRef<int64_t> shape,
                      mlir::MLIRContext *ctx, mlir::Value memRef,
                      mlir::Type elemType);

mlir::Value getDenseMatrixFromMemRef(mlir::Location,
                                     mlir::ConversionPatternRewriter&,
                                     mlir::Value memRef, mlir::Type);
