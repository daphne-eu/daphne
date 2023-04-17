#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

template <class BinaryOp, class IOp, class FOp>
class BinaryOpLowering final : public mlir::OpConversionPattern<BinaryOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

   public:
    BinaryOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
        : mlir::OpConversionPattern<BinaryOp>(typeConverter, ctx) {
        this->setDebugName("ScalarOpLowering");
    }

    mlir::LogicalResult convertEwScalar(
        BinaryOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();
        auto loc = op.getLoc();

        if (lhs.getType().template isa<mlir::FloatType>() &&
            rhs.getType().template isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<FOp>(op.getOperation(),
                                             adaptor.getOperands());
            return mlir::success();
        }

        Value castedLhs = this->typeConverter->materializeTargetConversion(
            rewriter, loc,
            rewriter.getIntegerType(
                adaptor.getRhs().getType().getIntOrFloatBitWidth()),
            ValueRange{adaptor.getLhs()});

        Value castedRhs = this->typeConverter->materializeTargetConversion(
            rewriter, loc,
            rewriter.getIntegerType(
                adaptor.getRhs().getType().getIntOrFloatBitWidth()),
            ValueRange{adaptor.getRhs()});

        Value binaryOp = rewriter.create<IOp>(loc, castedLhs, castedRhs);

        Value res = this->typeConverter->materializeSourceConversion(
            rewriter, loc, lhs.getType(), ValueRange{binaryOp});

        rewriter.replaceOp(op, res);
        return mlir::success();
    }

    mlir::LogicalResult matchAndRewrite(
        BinaryOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();

        // no matrix
        if (!lhs.getType().template isa<mlir::daphne::MatrixType>() &&
            !rhs.getType().template isa<mlir::daphne::MatrixType>())
            return convertEwScalar(op, adaptor, rewriter);

        // for now assume matrix is LHS and RHS is non matrix
        mlir::daphne::MatrixType lhsTensor =
            adaptor.getLhs()
                .getType()
                .template dyn_cast<mlir::daphne::MatrixType>();
        auto tensorType = lhsTensor.getElementType();
        auto lhsRows = lhsTensor.getNumRows();
        auto lhsCols = lhsTensor.getNumCols();
        auto lhsMemRefType =
            mlir::MemRefType::get({lhsRows, lhsCols}, tensorType);

        mlir::Type elementType{};
        mlir::Value memRefLhs =
            rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
                op->getLoc(), lhsMemRefType, adaptor.getLhs());
        mlir::Value memRefRhs{};
        bool isMatrixMatrix = rhs.getType().template isa<mlir::daphne::MatrixType>();

        if (isMatrixMatrix) {
            memRefRhs = rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
                op->getLoc(), lhsMemRefType, adaptor.getRhs());
            elementType = lhsMemRefType.getElementType();
        } else {
            elementType = rhs.getType();
        }

        mlir::Value outputMemRef =
            insertAllocAndDealloc(lhsMemRefType, op->getLoc(), rewriter);
        SmallVector<int64_t, 4> lowerBounds(/*Rank=*/2, /*Value=*/0);
        SmallVector<int64_t, 4> steps(/*Rank=*/2, /*Value=*/1);
        buildAffineLoopNest(
            rewriter, op.getLoc(), lowerBounds,
            {lhsTensor.getNumRows(), lhsTensor.getNumCols()}, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                mlir::Value loadLhs =
                    nestedBuilder.create<AffineLoadOp>(loc, memRefLhs, ivs);
                mlir::Value binaryOp{};

                // TODO(phil): cast only LHS/RHS in case only one of them is not
                // signless/float, currently expects LHS and RHS to be of same
                // type
                if (adaptor.getRhs().getType().template isa<mlir::FloatType>()) {
                    binaryOp =
                        nestedBuilder.create<FOp>(loc, loadLhs, adaptor.getRhs());

                    nestedBuilder.create<AffineStoreOp>(loc, binaryOp, outputMemRef,
                                                        ivs);
                    return;
                }

                mlir::Value rhs{};
                if (isMatrixMatrix)
                    rhs = nestedBuilder.create<AffineLoadOp>(loc, memRefRhs, ivs);
                else
                    rhs = adaptor.getRhs();

                if (elementType.isInteger(
                        // is integer
                        elementType.getIntOrFloatBitWidth())) {
                    Value castedLhs =
                        this->typeConverter->materializeTargetConversion(
                            nestedBuilder, loc,
                            nestedBuilder.getIntegerType(
                                lhsMemRefType.getElementTypeBitWidth()),
                            ValueRange{loadLhs});

                    Value castedRhs =
                        this->typeConverter->materializeTargetConversion(
                            nestedBuilder, loc,
                            nestedBuilder.getIntegerType(
                                lhsMemRefType.getElementTypeBitWidth()),
                            ValueRange{rhs});

                    binaryOp =
                        nestedBuilder.create<IOp>(loc, castedLhs, castedRhs);
                    Value castedRes =
                        this->typeConverter->materializeSourceConversion(
                            nestedBuilder, loc, elementType,
                            ValueRange{binaryOp});
                    nestedBuilder.create<AffineStoreOp>(loc, castedRes,
                                                        outputMemRef, ivs);
                } else {
                    // is float
                    binaryOp = nestedBuilder.create<FOp>(loc, loadLhs, rhs);
                    nestedBuilder.create<AffineStoreOp>(loc, binaryOp,
                                                        outputMemRef, ivs);
                }
            });
        mlir::Value output = getDenseMatrixFromMemRef(op->getLoc(), rewriter,
                                                      outputMemRef, op.getType());
        rewriter.replaceOp(op, output);
        return mlir::success();
    }
};

// clang-format off
using AddOpLowering = BinaryOpLowering<mlir::daphne::EwAddOp, mlir::arith::AddIOp, mlir::arith::AddFOp>;
using SubOpLowering = BinaryOpLowering<mlir::daphne::EwSubOp, mlir::arith::SubIOp, mlir::arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<mlir::daphne::EwMulOp, mlir::arith::MulIOp, mlir::arith::MulFOp>;
// using SqrtOpLowering = BinaryOpLowering<mlir::daphne::EwSqrtOp, mlir::math::SqrtOp, mlir::math::SqrtOp>;
//using DivOpLowering = ScalarOpLowering<mlir::daphne::EwDivOp, mlir::arith::DivFOp, mlir::arith::DivFOp>;
// // // TODO(phil): IPowIOp has been added to MathOps.td with 08b4cf3 Aug 10
// using PowOpLowering = ScalarOpLowering<mlir::daphne::EwPowOp, mlir::math::PowFOp, mlir::math::PowFOp>;
// // // TODO(phil): AbsIOp  has been added to MathOps.td with 7d9fc95 Aug 08
// using AbsOpLowering = ScalarOpLowering<mlir::daphne::EwAbsOp, mlir::math::AbsFOp, mlir::math::AbsFOp>;
// // // TODO(phil)
// using LnOpLowering = ScalarOpLowering<mlir::daphne::EwLnOp, mlir::math::LogOp, mlir::math::LogOp>;
// clang-format on

namespace {
struct LowerScalarOpsPass
    : public mlir::PassWrapper<LowerScalarOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit LowerScalarOpsPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry
            .insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                    mlir::memref::MemRefDialect, mlir::daphne::DaphneDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void LowerScalarOpsPass::runOnOperation() {
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

    target
        .addLegalDialect<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                         mlir::AffineDialect, mlir::LLVM::LLVMDialect,
                         mlir::daphne::DaphneDialect, mlir::BuiltinDialect>();

    // math::sqrt only supports floating point
    target.addDynamicallyLegalOp<mlir::daphne::EwSqrtOp>(
        [&](mlir::daphne::EwSqrtOp op) {
            return isa<mlir::IntegerType>(op.getArg().getType());
        });

    target.addDynamicallyLegalOp<mlir::daphne::EwAddOp, mlir::daphne::EwSubOp,
                                 mlir::daphne::EwMulOp>([](Operation *op) {
        if (op->getOperandTypes()[0].isa<mlir::daphne::MatrixType>() &&
            op->getOperandTypes()[1].isa<mlir::daphne::MatrixType>()) {
            mlir::daphne::MatrixType lhs =
                op->getOperandTypes()[0]
                    .template dyn_cast<mlir::daphne::MatrixType>();
            mlir::daphne::MatrixType rhs =
                op->getOperandTypes()[1]
                    .template dyn_cast<mlir::daphne::MatrixType>();
            if (lhs.getNumRows() != rhs.getNumRows() ||
                lhs.getNumCols() != rhs.getNumCols() ||
                lhs.getNumRows() == -1 || lhs.getNumCols() == -1)
                return true;

            return false;
        }

        if (op->getOperandTypes()[0].isa<mlir::daphne::MatrixType>()) {
            mlir::daphne::MatrixType lhsTensor =
                op->getOperandTypes()[0].dyn_cast<mlir::daphne::MatrixType>();
            return lhsTensor.getNumRows() == -1 || lhsTensor.getNumCols() == -1;
        }

        return false;
    });

    patterns.insert<AddOpLowering, SubOpLowering, MulOpLowering>(typeConverter,
                                                                 &getContext());

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerScalarOpsPass() {
    return std::make_unique<LowerScalarOpsPass>();
}
