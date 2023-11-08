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

#include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

class EwModOpLowering
    : public mlir::OpConversionPattern<mlir::daphne::EwModOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    [[nodiscard]] bool optimization_viable(mlir::Value divisor) const {
        std::pair<bool, int64_t> isConstant =
            CompilerUtils::isConstant<int64_t>(divisor);
        return isConstant.first && (isConstant.second & (isConstant.second - 1)) == 0;
    }

    void optimizeEwModOp(mlir::Value memRef, mlir::Value divisor,
                         ArrayRef<int64_t> shape,
                         ConversionPatternRewriter &rewriter,
                         Location loc) const {
        // divisor - 1
        mlir::Value cst_one = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));

        auto casted_divisor = typeConverter->materializeTargetConversion(
            rewriter, loc, rewriter.getI64Type(), ValueRange{divisor});

        mlir::Value rhs =
            rewriter.create<mlir::arith::SubIOp>(loc, casted_divisor, cst_one);

        SmallVector<int64_t, 4> lowerBounds(/*Rank=*/2, /*Value=*/0);
        SmallVector<int64_t, 4> steps(/*Rank=*/2, /*Value=*/1);
        buildAffineLoopNest(
            rewriter, loc, lowerBounds, shape, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                mlir::Value load =
                    nestedBuilder.create<AffineLoadOp>(loc, memRef, ivs);
                mlir::Value res{};

                Value castedLhs =
                    this->typeConverter->materializeTargetConversion(
                        nestedBuilder, loc,
                        nestedBuilder.getIntegerType(
                            divisor.getType().getIntOrFloatBitWidth()),
                        ValueRange{load});

                res = nestedBuilder.create<arith::AndIOp>(loc, castedLhs, rhs);
                Value castedRes =
                    this->typeConverter->materializeSourceConversion(
                        nestedBuilder, loc, divisor.getType(), ValueRange{res});

                nestedBuilder.create<AffineStoreOp>(loc, castedRes, memRef,
                                                    ivs);
            });
    }

    void lowerEwModOp(mlir::Value memRef, mlir::Value divisor,
                      ArrayRef<int64_t> shape,
                      ConversionPatternRewriter &rewriter, Location loc) const {
        SmallVector<int64_t, 4> lowerBounds(/*Rank=*/2, /*Value=*/0);
        SmallVector<int64_t, 4> steps(/*Rank=*/2, /*Value=*/1);
        buildAffineLoopNest(
            rewriter, loc, lowerBounds, shape, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                mlir::Value load =
                    nestedBuilder.create<AffineLoadOp>(loc, memRef, ivs);
                mlir::Value res{};

                // this is enough since divisor will be casted to float if
                // matrix is float
                if (divisor.getType().isa<mlir::FloatType>()) {
                    res =
                        nestedBuilder.create<arith::RemFOp>(loc, load, divisor);
                    nestedBuilder.create<AffineStoreOp>(loc, res, memRef, ivs);
                    return;
                }

                Value castedLhs =
                    this->typeConverter->materializeTargetConversion(
                        nestedBuilder, loc,
                        nestedBuilder.getIntegerType(
                            divisor.getType().getIntOrFloatBitWidth()),
                        ValueRange{load});

                Value castedRhs =
                    this->typeConverter->materializeTargetConversion(
                        nestedBuilder, loc,
                        nestedBuilder.getIntegerType(
                            divisor.getType().getIntOrFloatBitWidth()),
                        ValueRange{divisor});

                res = nestedBuilder.create<arith::RemSIOp>(loc, castedLhs,
                                                           castedRhs);
                Value castedRes =
                    this->typeConverter->materializeSourceConversion(
                        nestedBuilder, loc, divisor.getType(), ValueRange{res});

                nestedBuilder.create<AffineStoreOp>(loc, castedRes, memRef,
                                                    ivs);
            });
    }

    mlir::LogicalResult matchAndRewrite(
        mlir::daphne::EwModOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::daphne::MatrixType lhsTensor =
            adaptor.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();
        auto lhsRows = lhsTensor.getNumRows();
        auto lhsCols = lhsTensor.getNumCols();

        auto lhsMemRefType = mlir::MemRefType::get({lhsRows, lhsCols},
                                                   lhsTensor.getElementType());

        // daphne::Matrix -> memref
        mlir::Value lhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                op->getLoc(), lhsMemRefType, adaptor.getLhs());
        mlir::Value rhs = adaptor.getRhs();

        if (optimization_viable(rhs))
            optimizeEwModOp(lhs, rhs,
                            {lhsTensor.getNumRows(), lhsTensor.getNumCols()},
                            rewriter, op->getLoc());
        else
            lowerEwModOp(lhs, rhs,
                         {lhsTensor.getNumRows(), lhsTensor.getNumCols()},
                         rewriter, op->getLoc());

        mlir::Value output = convertMemRefToDenseMatrix(op->getLoc(), rewriter,
                                                        lhs, op.getType());
        rewriter.replaceOp(op, output);
        return success();
    }
};

namespace {
/**
 * @brief Performs an integer mod optimization on the EwModOp operator by
 * lowering to an affine loop structure and performing the mod op on values
 * loaded from a MemRef.
 *
 * If possible, we additionally perform the integer modulo optimization by
 * replacing the modulo with an bitwise AND and a subtraction.
 */
struct ModOpLoweringPass
    : public mlir::PassWrapper<ModOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit ModOpLoweringPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry
            .insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                    mlir::memref::MemRefDialect, mlir::daphne::DaphneDialect>();
    }
    void runOnOperation() final;

    StringRef getArgument() const final { return "lower-mod"; }
    StringRef getDescription() const final {
        return "Performs an integer mod optimization on the EwModOp operator "
               "by lowering to an affine loop structure"
               "and performing the mod op on values loaded from a MemRef.";
    }
};
}  // end anonymous namespace

void ModOpLoweringPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    typeConverter.addConversion(convertInteger);
    typeConverter.addConversion(convertFloat);
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addArgumentMaterialization(materializeCastFromIllegal);
    typeConverter.addSourceMaterialization(materializeCastToIllegal);
    typeConverter.addTargetMaterialization(materializeCastFromIllegal);

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::daphne::DaphneDialect>();

    target.addIllegalOp<mlir::daphne::EwModOp>();

    patterns.insert<EwModOpLowering>(typeConverter, &getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createModOpLoweringPass() {
    return std::make_unique<ModOpLoweringPass>();
}
