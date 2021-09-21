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

    std::unordered_map <std::string, mlir::Value> tables;
    struct SqlReplacement : public RewritePattern{

        SqlReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        // : RewritePattern(daphne::SqlOp::getOperationName(), benefit, context)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            std::stringstream callee;
            callee << op->getName().stripDialect().str();

            std::cout << "RewriteSqlOpPass matched:\t" << callee.str() << std::endl;

            if(callee.str() == "register"){
                mlir::daphne::RegisterOp rOp = static_cast<mlir::daphne::RegisterOp>(op);

                std::stringstream view_stream;
                view_stream << rOp.view().str();
                mlir::Value arg = rOp.arg();

                tables[view_stream.str()] = arg;
                rewriter.eraseOp(op);
                std::cout << "Erased Op" << std::endl;
                return success();
            }else if(callee.str() == "sql"){
                mlir::daphne::SqlOp sqlop = static_cast<mlir::daphne::SqlOp>(op);
                std::cout << sqlop.sql().str() << std::endl;

                // auto moduleOp = ModuleOp::create(rewriter.getUnknownLoc());
                // auto *body = moduleOp.getBody();
                // rewriter.setInsertionPoint(body, body->begin());

                std::stringstream sql_query;
                sql_query << sqlop.sql().str();

                DaphneSQLParser parser;
                parser.setView(tables);
                mlir::Value result_op = parser.parseStreamFrame(rewriter, sql_query);
                // moduleOp->dump();

                rewriter.replaceOp(op, result_op);
                std::cout << "Rewriten Op" << std::endl;
                return success();
            }
            return failure();
        }
    };

    struct RewriteSqlOpPass
    : public PassWrapper <RewriteSqlOpPass, OperationPass<ModuleOp>>
    {
        void runOnOperation() final;
    };
}

void RewriteSqlOpPass::runOnOperation()
{
//    std::cout << "Start RewriteSqlOpPass" << std::endl;
    auto module = getOperation();

    OwningRewritePatternList patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addIllegalOp<mlir::daphne::SqlOp, mlir::daphne::RegisterOp>();

    patterns.insert<SqlReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
//    std::cout << "End RewriteSqlOpPass" << std::endl;

}

std::unique_ptr<Pass> daphne::createRewriteSqlOpPass()
{
    return std::make_unique<RewriteSqlOpPass>();
}
