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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

template <class BinaryOp, class IOp, class FOp>
class ScalarOpLowering final : public mlir::OpConversionPattern<BinaryOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

   public:
    ScalarOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
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

        mlir::Value memRef =
            rewriter.create<mlir::daphne::GetMemRefDenseMatrix>(
                op->getLoc(), lhsMemRefType, adaptor.getLhs());

        SmallVector<int64_t, 4> lowerBounds(/*Rank=*/2, /*Value=*/0);
        SmallVector<int64_t, 4> steps(/*Rank=*/2, /*Value=*/1);
        buildAffineLoopNest(
            rewriter, op.getLoc(), lowerBounds,
            {lhsTensor.getNumRows(), lhsTensor.getNumCols()}, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                mlir::Value load =
                    nestedBuilder.create<AffineLoadOp>(loc, memRef, ivs);
                mlir::Value binaryOp{};

                // TODO(phil): cast only LHS/RHS in case only one of them is not
                // signless/float, currently expects LHS and RHS to be of same
                // type
                if (rhs.getType().template isa<mlir::FloatType>()) {
                    binaryOp =
                        nestedBuilder.create<FOp>(loc, load, adaptor.getRhs());

                    nestedBuilder.create<AffineStoreOp>(loc, binaryOp, memRef,
                                                        ivs);
                    return;
                }

                Value castedLhs =
                    this->typeConverter->materializeTargetConversion(
                        nestedBuilder, loc,
                        nestedBuilder.getIntegerType(
                            adaptor.getRhs().getType().getIntOrFloatBitWidth()),
                        ValueRange{load});

                Value castedRhs =
                    this->typeConverter->materializeTargetConversion(
                        nestedBuilder, loc,
                        nestedBuilder.getIntegerType(
                            adaptor.getRhs().getType().getIntOrFloatBitWidth()),
                        ValueRange{adaptor.getRhs()});

                binaryOp = nestedBuilder.create<IOp>(loc, castedLhs, castedRhs);
                Value castedRes =
                    this->typeConverter->materializeSourceConversion(
                        nestedBuilder, loc, adaptor.getRhs().getType(),
                        ValueRange{binaryOp});
                nestedBuilder.create<AffineStoreOp>(loc, castedRes, memRef,
                                                    ivs);
            });
        mlir::Value output = getDenseMatrixFromMemRef(op->getLoc(), rewriter,
                                                      memRef, op.getType());
        rewriter.replaceOp(op, output);
        return mlir::success();
    }
};

using AddOpLowering =
    ScalarOpLowering<mlir::daphne::EwAddOp, mlir::arith::AddIOp,
                     mlir::arith::AddFOp>;
// using SubOpLowering =
// ScalarOpLowering<mlir::daphne::EwSubOp, mlir::arith::SubIOp,
// mlir::arith::SubFOp>; using MulOpLowering =
// ScalarOpLowering<mlir::daphne::EwMulOp, mlir::arith::MulIOp,
// mlir::arith::MulFOp>; using DivOpLowering =
// ScalarOpLowering<mlir::daphne::EwDivOp, mlir::arith::DivFOp,
// mlir::arith::DivFOp>;
// // // TODO(phil): IPowIOp has been added to MathOps.td with 08b4cf3 Aug 10
// using PowOpLowering = ScalarOpLowering<mlir::daphne::EwPowOp,
// mlir::math::PowFOp, mlir::math::PowFOp>;
// // // TODO(phil): AbsIOp  has been added to MathOps.td with 7d9fc95 Aug 08
// using AbsOpLowering = ScalarOpLowering<mlir::daphne::EwAbsOp,
// mlir::math::AbsFOp, mlir::math::AbsFOp>;
// // // TODO(phil)
// using LnOpLowering = ScalarOpLowering<mlir::daphne::EwLnOp,
// mlir::math::LogOp, mlir::math::LogOp>;

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

// TODO(phil): move to LoweringUtils
Type convertFloat(mlir::FloatType floatType) {
    return IntegerType::get(floatType.getContext(),
                            floatType.getIntOrFloatBitWidth());
}

Type convertInteger(mlir::IntegerType intType) {
    return IntegerType::get(intType.getContext(),
                            intType.getIntOrFloatBitWidth());
}

llvm::Optional<Value> materializeCastFromIllegal(OpBuilder &builder, Type type,
                                                 ValueRange inputs,
                                                 Location loc) {
    Type fromType = getElementTypeOrSelf(inputs[0].getType());
    Type toType = getElementTypeOrSelf(type);

    if ((!fromType.isSignedInteger() && !fromType.isUnsignedInteger()) ||
        !toType.isSignlessInteger())
        return std::nullopt;
    // Use unrealized conversion casts to do signful->signless conversions.
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
}

llvm::Optional<Value> materializeCastToIllegal(OpBuilder &builder, Type type,
                                               ValueRange inputs,
                                               Location loc) {
    Type fromType = getElementTypeOrSelf(inputs[0].getType());
    Type toType = getElementTypeOrSelf(type);

    if (!fromType.isSignlessInteger() ||
        (!toType.isSignedInteger() && !toType.isUnsignedInteger()))
        return std::nullopt;
    // Use unrealized conversion casts to do signless->signful conversions.
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
}

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

    target.addIllegalOp<mlir::daphne::EwAddOp, mlir::daphne::EwSubOp,
                        mlir::daphne::EwMulOp>();

    // patterns.insert<AddOpLowering, SubOpLowering, MulOpLowering,
    // DivOpLowering,
    //              PowOpLowering, AbsOpLowering>(typeConverter, &getContext());
    patterns.insert<AddOpLowering>(typeConverter, &getContext());

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerScalarOpsPass() {
    return std::make_unique<LowerScalarOpsPass>();
}
