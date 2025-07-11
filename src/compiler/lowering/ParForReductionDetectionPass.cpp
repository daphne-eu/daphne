#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/RegionUtils.h>
#include "mlir/IR/Attributes.h"

using namespace mlir;

namespace {
struct ParForReductionDetectionPass : public PassWrapper<ParForReductionDetectionPass, OperationPass<func::FuncOp>> {
    void runOnOperation() final {
        func::FuncOp func = getOperation();
        func->walk([&](daphne::ParForOp parForOp) { detectReduction(parForOp); });
        //exit(-1);
    }

    func::FuncOp buildCombinerFunction(Operation *ctxOp, Value resultVal, Operation *rootCombOp, ModuleOp module,
                                       size_t idx, ArrayRef<Operation *> combinerOps) {
        OpBuilder builder(module);

        llvm::SetVector<Value> externalInputs;
        for (Operation *op : combinerOps) {
            for (Value operand : op->getOperands()) {
                if (auto defOp = operand.getDefiningOp()) {
                    if (!llvm::is_contained(combinerOps, defOp))
                        externalInputs.insert(operand);
                } else if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
                    externalInputs.insert(blockArg);
                }
            }
        }

        SmallVector<Type> argTypes;
        for (Value v : externalInputs)
            argTypes.push_back(v.getType());

        std::stringstream name;
        
        static auto combIdx = 0;
        name << "parFor" << combIdx++ << "_combiner";
        
        auto funcType = builder.getFunctionType(argTypes, {resultVal.getType()});
        auto combinerFunc = builder.create<func::FuncOp>(ctxOp->getLoc(), name.str(), funcType);

        Block *entry = combinerFunc.addEntryBlock();
        IRMapping mapping;
        for (auto [i, val] : llvm::enumerate(externalInputs))
            mapping.map(val, entry->getArgument(i));

        builder.setInsertionPointToStart(entry);

        // Clone all combiner operations
        for (Operation *op : combinerOps) {
            builder.clone(*op, mapping);
        }

        Value resultValMapped = mapping.lookup(resultVal);
        builder.create<func::ReturnOp>(ctxOp->getLoc(), resultValMapped);
        return combinerFunc;
    }

    void detectReduction(daphne::ParForOp parForOp) {
        auto returnOp = llvm::dyn_cast<daphne::ReturnOp>(parForOp.getBodyStmt().back().getTerminator());
        if (!returnOp || returnOp.getNumOperands() == 0) {
            return; // No operands to analyze, i.e. there is no reduction to conduct after parellel computation
        }
        auto results = returnOp.getOperands();

        Block &loopBlock = parForOp.getBodyStmt().getBlocks().front();
        auto args = loopBlock.getArguments();
        // exclude induction variable and context arguments, since those are per definition not loop-carried variables
        auto maybeLoopCarried = args.drop_front(1).drop_back(1);
        llvm::errs() << "Analyzing ParForOp for reduction detection: \n";
        llvm::DenseMap<size_t, StringRef> mapping;
        OpBuilder builder(parForOp);
        for (auto [idx, arg] : llvm::enumerate(maybeLoopCarried)) {
            // corresponding return value
            Value *returnVal = nullptr;
            // if the argument is loop-carried, we can assume that it is a reduction variable
            if (!isLoopCarried(arg, returnOp, returnVal) && returnVal != nullptr)
                continue;

            llvm::errs() << "Found loop-carried argument: " << idx << "\n";
            // we analyze the use-def chain of the argument backwards from the corresponding return value to find out
            // which reduction should be applied
            SmallVector<Operation *> combinerOps;
            Operation *curOp = returnVal->getDefiningOp();
            //TODO: Check if there is side-effects (memory access/global state) and throw compilation error 
            while (curOp) {
                combinerOps.push_back(curOp);
                if (llvm::any_of(curOp->getOperands(), [&](Value operand) { return operand == arg; }))
                    break;
                // Follow the value further back
                bool foundNext = false;
                for (Value operand : curOp->getOperands()) {
                    if (auto defOp = operand.getDefiningOp()) {
                        curOp = defOp;
                        foundNext = true;
                        break;
                    }
                }
                if (!foundNext)
                    break;
            }
            // idk, how to handle mutliple combiners: affine does it the same way (see LoopAnalysis.cpp)
            if (combinerOps.empty() || combinerOps.size() != 1)
                continue;
            
            // DEBUG PRINT OUT 
            for (auto combinerOp : combinerOps) {
                llvm::errs() << "Combiner operation: " << *combinerOp << "\n";
            }
            Operation *root = combinerOps.back();
            // DEBUG PRINT 
            if (llvm::isa<daphne::InsertRowOp>(root)) {
                llvm::errs() << "Found reduction for argument " << idx << " in ParForOp: InsertRowOp\n";
            } else if (llvm::isa<daphne::MatMulOp>(root)) {
                llvm::errs() << "Found reduction for argument " << idx << " in ParForOp: MatMulOP\n";
            }

            auto combiner = buildCombinerFunction(parForOp, *returnVal, root, parForOp->getParentOfType<ModuleOp>(), idx, combinerOps);
            // Add function name to the mapping with according result idx 
            mapping[idx] = combiner.getSymName();
            
            llvm::errs() << "Built combiner function: " << combiner.getSymName();
        }
        // Map to attrbiutes, TODO: there is probably better way to do this 
        llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
        for (const auto &kv : mapping) {
            std::string key = std::to_string(kv.first);
            mlir::StringAttr name = builder.getStringAttr(key);
            mlir::StringAttr value = builder.getStringAttr(kv.second);
            attrs.emplace_back(name, value);
        }
        // Append result_idx:combiner_func_name mapping to the parfor 
        auto newDict = mlir::DictionaryAttr::get(builder.getContext(), attrs);
        parForOp.setCombinerMappingAttr(newDict);
        // ANOTHER ONE DEBUG PRINT 
        parForOp->getParentOfType<ModuleOp>().dump();
    }

    bool isLoopCarried(BlockArgument arg, daphne::ReturnOp returnOp, Value *&returnVal) {
        SetVector<Operation *> slice;
        mlir::getForwardSlice(arg, &slice);
        for (auto op : returnOp.getOperands()) {
            if (Operation *defOp = op.getDefiningOp()) {
                if (slice.contains(defOp)) {
                    returnVal = &op;
                    return true;
                }
            }
        }
        return false;
    }
};

} // end anonymous namespace

std::unique_ptr<Pass> daphne::createParForReductionDetectionPass() {
    return std::make_unique<ParForReductionDetectionPass>();
}
