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

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include <memory>
#include <utility>
#include <vector>

using namespace mlir;

#if 0
// At the moment, all of these operations are lowered to kernel calls.
template <typename BinaryOp, typename ReplIOp, typename ReplFOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp>
{
    using OpConversionPattern<BinaryOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(BinaryOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        Type type = op.getType();
        if (type.isa<IntegerType>()) {
            rewriter.replaceOpWithNewOp<ReplIOp>(op.getOperation(), operands);
        }
        else if (type.isa<FloatType>()) {
            rewriter.replaceOpWithNewOp<ReplFOp>(op.getOperation(), operands);
        }
        else {
            return failure();
        }
        return success();
    }
};
using AddOpLowering = BinaryOpLowering<daphne::AddOp, AddIOp, AddFOp>;
using SubOpLowering = BinaryOpLowering<daphne::SubOp, SubIOp, SubFOp>;
using MulOpLowering = BinaryOpLowering<daphne::MulOp, MulIOp, MulFOp>;
#endif

struct ReturnOpLowering : public OpRewritePattern<daphne::ReturnOp>
{
    using OpRewritePattern<daphne::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::ReturnOp op,
                                  PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<ReturnOp>(op, op.getOperands());
        return success();
    }
};

/// ConstantOp lowering for types not handled before (str)

class ConstantOpLowering : public OpConversionPattern<daphne::ConstantOp>
{
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::ConstantOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<ConstantOp>(op.getOperation(), op.value());
        return success();
    }
};

class CallKernelOpLowering : public OpConversionPattern<daphne::CallKernelOp>
{

    static std::vector<Type> getLLVMInputOutputTypes(Location &loc,
                                                     MLIRContext *context,
                                                     TypeConverter *typeConverter,
                                                     TypeRange resultTypes,
                                                     TypeRange operandTypes)
    {

        llvm::SmallVector<Type, 5> args;
        for (auto type : resultTypes) {
            if (typeConverter->isLegal(type)) {
                args.push_back(type);
            }
            else if (failed(typeConverter->convertType(type, args)))
                emitError(loc) << "Couldn't convert result type `" << type << "`\n";
        }
        for (auto type : operandTypes) {
            if (typeConverter->isLegal(type)) {
                args.push_back(type);
            }
            else if (failed(typeConverter->convertType(type, args)))
                emitError(loc) << "Couldn't convert operand type `" << type << "`\n";
        }
        
        std::vector<Type> argsLLVM;
        for (size_t i = 0; i < args.size(); i++) {
            Type type = args[i]; //.cast<Type>();
            if (i < resultTypes.size()) {
                // outputs have to be given by reference
                type = LLVM::LLVMPointerType::get(type);
            }

            argsLLVM.push_back(type);
        }
        return argsLLVM;
    }

    static FlatSymbolRefAttr
    getOrInsertFunctionAttr(OpBuilder &rewriter, ModuleOp module,
                            llvm::StringRef funcName,
                            LLVM::LLVMFunctionType llvmFnType)
    {
        auto *context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
            return SymbolRefAttr::get(context, funcName);

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, llvmFnType);
        return SymbolRefAttr::get(context, funcName);
    }

    static LLVM::LLVMFunctionType
    getKernelFuncSignature(MLIRContext *context, std::vector<Type> argsLLVM)
    {
        return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), argsLLVM,
                                           false);
    }

public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::CallKernelOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        auto module = op->getParentOfType<ModuleOp>();
        auto loc = op.getLoc();

        auto inputOutputTypes = getLLVMInputOutputTypes(
                                                        loc, rewriter.getContext(), typeConverter,
                                                        op.getResultTypes(), ValueRange(operands).getTypes());

        // create function protoype and get `FlatSymbolRefAttr` to it
        auto kernelRef = getOrInsertFunctionAttr(
                                                 rewriter, module, op.getCalleeAttr().getValue(),
                                                 getKernelFuncSignature(rewriter.getContext(), inputOutputTypes));

        auto kernelOperands =
                allocOutputReferences(loc, rewriter, operands, inputOutputTypes);

        // call function
        rewriter.create<CallOp>(
                loc, kernelRef,
                /*no return value*/ LLVM::LLVMVoidType::get(rewriter.getContext()),
                kernelOperands);
        rewriter.replaceOp(op, dereferenceOutputs(loc, rewriter, module,
                                                  operands.size(), kernelOperands));
        return success();
    }

private:

    static std::vector<Value>
    dereferenceOutputs(Location &loc, PatternRewriter &rewriter, ModuleOp &module,
                       size_t numInputs, std::vector<Value> kernelOperands)
    {
        // transformed results
        std::vector<Value> results;
        for (size_t i = 0; i < kernelOperands.size() - numInputs; i++) {
            // dereference output
            auto value = kernelOperands[i];
            // load element (dereference)
            auto resultVal = rewriter.create<LLVM::LoadOp>(loc, value);

            results.push_back(resultVal);
        }
        return results;
    }

    std::vector<Value>
    allocOutputReferences(Location &loc, PatternRewriter &rewriter,
                          ArrayRef<Value> operands,
                          std::vector<Type> inputOutputTypes) const
    {
        // constant of 1 for alloca of output
        Value cst1 =
                rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(1));

        std::vector<Value> kernelOperands;
        for (size_t i = 0; i < inputOutputTypes.size() - operands.size(); i++) {
            auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, inputOutputTypes[i], cst1);
            kernelOperands.push_back(allocaOp);

            // If the type of this result parameter is a pointer (i.e. when it
            // represents a matrix or frame), then initialize the allocated
            // element with a null pointer (required by the kernels). Otherwise
            // (i.e. when it represents a scalar), initialization is not
            // required.
            if(inputOutputTypes[i].dyn_cast<LLVM::LLVMPointerType>().getElementType().isa<LLVM::LLVMPointerType>())
                rewriter.create<LLVM::StoreOp>(
                        loc,
                        rewriter.create<LLVM::NullOp>(
                                loc, LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 1))
                        ),
                        allocaOp
                );
        }
        for(auto op : operands)
            kernelOperands.push_back(op);
        return kernelOperands;
    }
};

/**
 * @brief Rewrites `daphne::CreateVariadicPackOp` to `LLVM::AllocaOp` to create
 * an array for the required number of occurrences of a variadic operand.
 */
class CreateVariadicPackOpLowering : public OpConversionPattern<daphne::CreateVariadicPackOp>
{
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::CreateVariadicPackOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
                op.getOperation(),
                LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(rewriter.getI1Type())),
                rewriter.create<ConstantOp>(op->getLoc(), op.numElementsAttr()),
                1
        );
        return success();
    }
};

/**
 * @brief Rewrites `daphne::StoreVariadicPackOp` to `LLVM::StoreOp` to store
 * an occurrence of a variadic operand to the respective position in an array
 * created by lowering `daphne::CreateVariadicPackOp`.
 */
class StoreVariadicPackOpLowering : public OpConversionPattern<daphne::StoreVariadicPackOp>
{
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::StoreVariadicPackOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        std::vector<Value> indices = {
            rewriter.create<ConstantOp>(op->getLoc(), op.posAttr())
        };
        auto addr = rewriter.create<LLVM::GEPOp>(
                op->getLoc(),
                operands[0].getType(),
                operands[0],
                indices
        );
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
                op.getOperation(),
                operands[1],
                addr
        );
        return success();
    }
};

namespace
{
    struct DaphneLowerToLLVMPass
    : public PassWrapper<DaphneLowerToLLVMPass, OperationPass<ModuleOp>>
    {

        void getDependentDialects(DialectRegistry & registry) const override
        {
            registry.insert<LLVM::LLVMDialect/*, scf::SCFDialect*/>();
        }
        void runOnOperation() final;
    };
} // end anonymous namespace

void DaphneLowerToLLVMPass::runOnOperation()
{
    auto module = getOperation();

    OwningRewritePatternList patterns(&getContext());

    LLVMTypeConverter typeConverter(&getContext());
    typeConverter.addConversion([&](daphne::MatrixType t)
    {
        return LLVM::LLVMPointerType::get(
                IntegerType::get(t.getContext(), 1));
    });
    typeConverter.addConversion([&](daphne::FrameType t)
    {
        return LLVM::LLVMPointerType::get(
                IntegerType::get(t.getContext(), 1));
    });
    typeConverter.addConversion([&](daphne::VariadicPackType t)
    {
        return LLVM::LLVMPointerType::get(
                LLVM::LLVMPointerType::get(
                IntegerType::get(t.getContext(), 1)));
    });

    LLVMConversionTarget target(getContext());

    // populate dialect conversions
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    target.addLegalOp<ModuleOp>();

    patterns.insert<CallKernelOpLowering>(typeConverter, &getContext());
    patterns.insert<
            ConstantOpLowering,
            ReturnOpLowering,
            CreateVariadicPackOpLowering,
            StoreVariadicPackOpLowering
    >(&getContext());

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createLowerToLLVMPass()
{
    return std::make_unique<DaphneLowerToLLVMPass>();
}
