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
#include <parser/sql/morphstore/MorphStoreSQLParser.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <string>
#include <unordered_map>

using namespace mlir;

namespace
{

    struct SqlReplacement : public RewritePattern {
        static std::unordered_map <std::string, mlir::Value> tables;

        SqlReplacement(MLIRContext * context, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
            /// registerView(...);
            if(auto regViewOp = llvm::dyn_cast<mlir::daphne::RegisterViewOp>(op)){
                std::stringstream view_stream;
                view_stream << regViewOp.view().str();
                mlir::Value arg = regViewOp.arg();

                tables[view_stream.str()] = arg;
                rewriter.eraseOp(op);
                return success();
            }
            
            /// sql(...);
            if(auto sqlOp = llvm::dyn_cast<mlir::daphne::SqlOp>(op)){
                std::stringstream sql_query;
                sql_query << sqlOp.sql().str();

//                #ifndef USE_MORPHSTORE
                SQLParser parser;
//                #else
//                MorphStoreSQLParser parser;
//                #endif
                parser.setView(tables);
                std::string sourceName;
                llvm::raw_string_ostream ss(sourceName);
                ss << "[sql query @ " << sqlOp->getLoc() << ']';
                mlir::Value result_op = parser.parseStreamFrame(rewriter, sql_query, sourceName);

                rewriter.replaceOp(op, result_op);
                return success();
            }
            return failure();
        }
    };
    
    SQLParser::viewType SqlReplacement::tables = {};

    struct RewriteSqlOpPass
    : public PassWrapper <RewriteSqlOpPass, OperationPass<ModuleOp>>
    {
        void runOnOperation() final;
    };
}

void RewriteSqlOpPass::runOnOperation()
{
    auto module = getOperation();

    OwningRewritePatternList patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addIllegalOp<mlir::daphne::SqlOp, mlir::daphne::RegisterViewOp>();

    patterns.insert<SqlReplacement>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteSqlOpPass()
{
    return std::make_unique<RewriteSqlOpPass>();
}
