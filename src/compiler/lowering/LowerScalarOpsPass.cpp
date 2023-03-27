#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "compiler/utils/CompilerUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"



#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <vector>
#include <iostream>

// template <typename BinaryOp, typename IOp, typename FOp>
struct ScalarOpLowering : public mlir::OpConversionPattern<mlir::daphne::EwAddOp> {
    public:
    using mlir::OpConversionPattern<mlir::daphne::EwAddOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::daphne::EwAddOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {

        mlir::Type type = op.getType();

        if (type.isa<mlir::IntegerType>()) {
            rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op.getOperation(), adaptor.getOperands());
        } else if (type.isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op.getOperation(), adaptor.getOperands());
        } else {
            return mlir::failure();
        }
        return mlir::success();
    }
};
// using AddOpLowering = ScalarOpLowering<mlir::daphne::EwSubOp, mlir::arith::AddIOp, mlir::arith::AddFOp>;
// using SubOpLowering = ScalarOpLowering<mlir::daphne::EwSubOp, mlir::arith::SubIOp, mlir::arith::SubFOp>;
// using MulOpLowering = ScalarOpLowering<mlir::daphne::EwMulOp, mlir::arith::MulIOp, mlir::arith::MulFOp>;
// using DivOpLowering = ScalarOpLowering<mlir::daphne::EwDivOp, mlir::arith::DivFOp, mlir::arith::DivFOp>;
// // TODO(phil): IPowIOp has been added to MathOps.td with 08b4cf3 Aug 10
// using PowOpLowering = ScalarOpLowering<mlir::daphne::EwPowOp, mlir::math::PowFOp, mlir::math::PowFOp>;
// // TODO(phil): AbsIOp  has been added to MathOps.td with 7d9fc95 Aug 08
// using AbsOpLowering = ScalarOpLowering<mlir::daphne::EwAbsOp, mlir::math::AbsFOp, mlir::math::AbsFOp>;
// // TODO(phil)
// using LnOpLowering = ScalarOpLowering<mlir::daphne::EwLnOp, mlir::math::LogOp, mlir::math::LogOp>;



namespace
{
    struct LowerScalarOpsPass
    : public mlir::PassWrapper<LowerScalarOpsPass, mlir::OperationPass<mlir::ModuleOp>>
    {
		explicit LowerScalarOpsPass() { }

        void getDependentDialects(mlir::DialectRegistry & registry) const override
        {
            registry.insert<mlir::LLVM::LLVMDialect/*, scf::SCFDialect*/>();
        }
        void runOnOperation() final;
    };
} // end anonymous namespace

void LowerScalarOpsPass::runOnOperation()
{
    auto module = getOperation();
    mlir::RewritePatternSet patterns(&getContext());

    mlir::ConversionTarget target(getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalOp<mlir::daphne::EwAddOp, mlir::daphne::EwSubOp, mlir::daphne::EwMulOp>();

    // patterns.insert<AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
    //              PowOpLowering, AbsOpLowering>(typeConverter, &getContext());
    patterns.insert<ScalarOpLowering>(typeConverter, &getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerScalarOpsPass()
{
    return std::make_unique<LowerScalarOpsPass>();
}
