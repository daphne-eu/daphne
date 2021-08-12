#include <parser/daphnesql/DaphneSQLParser.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace mlir;

namespace
{
    struct AlgebraReplacement : public RewritePattern
    {

        AlgebraReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        // : RewritePattern(daphne::SqlOp::getOperationName(), benefit, context)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            return success();
        }
    };


    struct LowerRelationalAlgebraToDaphneOpPass
    : public PassWrapper <LowerRelationalAlgebraToDaphneOpPass, OperationPass<ModuleOp>>
    {
        void runOnOperation() final;
    };

}

void LowerRelationalAlgebraToDaphneOpPass::runOnOperation()
{
    // std::cout << "Start RewriteSqlOpPass" << std::endl;
    // auto module = getOperation();
    //
    // OwningRewritePatternList patterns(&getContext());
    // //
    // // // convert other operations
    // ConversionTarget target(getContext());
    // target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    // target.addLegalOp<ModuleOp, FuncOp>();
    // target.addIllegalOp<mlir::daphne::SqlOp, mlir::daphne::RegisterOp>();
    // //
    // patterns.insert<AlgebraReplacement>(&getContext());
    // //
    // if (failed(applyPartialConversion(module, target, std::move(patterns))))
    //     signalPassFailure();
    // std::cout << "End RewriteSqlOpPass" << std::endl;

}

std::unique_ptr<Pass> daphne::createLowerRelationalAlgebraToDaphneOpPass(){
    return std::make_unique<LowerRelationalAlgebraToDaphneOpPass>();
}
