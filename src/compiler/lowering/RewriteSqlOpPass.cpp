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
#include <parser/sql/SQLParser.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            if(auto rOp = llvm::dyn_cast<mlir::daphne::RegisterViewOp>(op)){
                std::stringstream view_stream;
                view_stream << rOp.getView().str();
                mlir::Value arg = rOp.getArg();

                tables[view_stream.str()] = arg;
                rewriter.eraseOp(op);
                return success();
            }else if(auto sqlop = llvm::dyn_cast<mlir::daphne::SqlOp>(op)){
                std::stringstream sql_query;
                sql_query << sqlop.getSql().str();

                SQLParser parser;
                parser.setView(tables);
                std::string sourceName;
                llvm::raw_string_ostream ss(sourceName);
                ss << "[sql query @ " << sqlop->getLoc() << ']';
                mlir::Value result_op;
                try {
                    result_op = parser.parseStreamFrame(rewriter, sql_query, sourceName);
                }
                catch (std::runtime_error& re) {
                    spdlog::error("Final catch std::runtime_error in {}:{}: \n{}",__FILE__, __LINE__, re.what());
                    return failure();
                }
                rewriter.replaceOp(op, result_op);
                // TODO Why is this necessary when we have already replaced the op?
                rewriter.replaceAllUsesWith(op->getResult(0), result_op);
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
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    target.addIllegalOp<mlir::daphne::SqlOp, mlir::daphne::RegisterViewOp>();

    patterns.add<SqlReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteSqlOpPass()
{
    return std::make_unique<RewriteSqlOpPass>();
}
