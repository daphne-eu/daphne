#include <parser/sql/SQLParser.h>
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

}

std::unique_ptr<Pass> daphne::createLowerRelationalAlgebraToDaphneOpPass(){
    return std::make_unique<LowerRelationalAlgebraToDaphneOpPass>();
}
