#include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "llvm/Support/Debug.h"
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

#define DEBUG_TYPE "dm-opt"

using namespace mlir;

class IntegerModOpt : public mlir::OpConversionPattern<mlir::daphne::EwModOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    [[nodiscard]] static bool optimization_viable(mlir::daphne::EwModOp op) {
        if (!op.getRhs().getType().isUnsignedInteger()) return false;

        std::pair<bool, uint64_t> isConstant =
            CompilerUtils::isConstant<uint64_t>(op.getRhs());
        // Apply (lhs % rhs) to (lhs & (rhs - 1)) optimization when rhs is a power of two
        return isConstant.first && (isConstant.second & (isConstant.second - 1)) == 0;
    }

    mlir::LogicalResult matchAndRewrite(
        mlir::daphne::EwModOp op, OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value cst_one = rewriter.create<mlir::daphne::ConstantOp>(
            op.getLoc(), static_cast<uint64_t>(1));
        mlir::Value sub = rewriter.create<mlir::daphne::EwSubOp>(
            op.getLoc(), adaptor.getRhs(), cst_one);
        mlir::Value andOp = rewriter.create<mlir::daphne::EwBitwiseAndOp>(
            op.getLoc(), adaptor.getLhs(), sub);
        rewriter.replaceOp(op, andOp);
        return success();
    }
};

namespace {
/**
 * @brief This pass transforms operations (currently limited to the EwModOp) in
 * the DaphneDialect to a different set of operations also from the
 * DaphneDialect.
 */
struct DenseMatrixOptPass
    : public mlir::PassWrapper<DenseMatrixOptPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit DenseMatrixOptPass() {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                        mlir::daphne::DaphneDialect>();
    }
    void runOnOperation() final;

    StringRef getArgument() const final { return "opt-daphne"; }
    StringRef getDescription() const final {
        return "Performs optimizations on the DaphneIR by transforming "
               "operations in the DaphneDialect to a set of other operation "
               "also from the DaphneDialect.";
    }
};
}  // end anonymous namespace

void DenseMatrixOptPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    typeConverter.addConversion([](Type type) { return type; });

    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::daphne::DaphneDialect>();

    target.addDynamicallyLegalOp<mlir::daphne::EwModOp>(
        [&](mlir::daphne::EwModOp op) {
            return !IntegerModOpt::optimization_viable(op);
        });

    patterns.insert<IntegerModOpt>(typeConverter, &getContext());

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createDaphneOptPass() {
    return std::make_unique<DenseMatrixOptPass>();
}
