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
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context)
        {}

        LogicalResult matchAndRewrite(
            Operation *op,
            PatternRewriter &rewriter
        ) const override
        {
            if(auto rOp = llvm::dyn_cast<mlir::daphne::RegisterViewOp>(op)){
                std::stringstream view_stream;
                view_stream << rOp.view().str();
                mlir::Value arg = rOp.arg();

                tables[view_stream.str()] = arg;
                rewriter.eraseOp(op);
                return success();
            }else if(auto sqlop = llvm::dyn_cast<mlir::daphne::SqlOp>(op)){

#ifndef USE_DUCKDB
                std::stringstream sql_query;
                sql_query << sqlop.sql().str();

                SQLParser parser;
                parser.setView(tables);
                std::string sourceName;
                llvm::raw_string_ostream ss(sourceName);
                ss << "[sql query @ " << sqlop->getLoc() << ']';
                mlir::Value result_op = parser.parseStreamFrame(rewriter, sql_query, sourceName);

                rewriter.replaceOp(op, result_op);
                return success();
#else

                mlir::Location loc = op->getLoc();

                //TODO Pass error: some operation has an unknown result type, but does not implement the type inference interface: daphne.duckdbsql
                //Create type inferance and ask an oracle what type the result will be.
                mlir::Type t = mlir::daphne::UnknownType::get(rewriter.getContext());
                std::vector<mlir::Type> colT;
                colT.push_back(t);
                mlir::Type rt = mlir::daphne::FrameType::get(rewriter.getContext(), colT);

                std::vector<mlir::Value> duckTables;
                std::vector<mlir::Value> duckTableNames;
                mlir::Value query = rewriter.create<mlir::daphne::ConstantOp>(loc, sqlop.sql().str());

                for( const auto& [key, value] : tables ) {
                    duckTables.push_back(value);
                    std::cout << "A table is called: "<<  key << std::endl;
                    mlir::Value name = rewriter.create<mlir::daphne::ConstantOp>(loc, key);
                    duckTableNames.push_back(name);
                }

                mlir::Value replacementOp = rewriter.create<daphne::DuckDbSqlOp>(op->getLoc(), rt, query, duckTables, duckTableNames);
                rewriter.replaceOp(op, replacementOp);
                std::cout << "WOWIE!!!!" << std::endl;
                return success();
#endif
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
