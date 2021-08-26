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
#include "mlir/IR/BlockAndValueMapping.h"

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

// TODO We might want to merge this with ValueTypeUtils, and maybe place it
// somewhere central.
std::string mlirTypeToCppTypeName(Type t, bool generalizeToStructure) {
    if(t.isF64())
        return "double";
    else if(t.isF32())
        return "float";
    else if(t.isSignedInteger(8))
        return "int8_t";
    else if(t.isSignedInteger(32))
        return "int32_t";
    else if(t.isSignedInteger(64))
        return "int64_t";
    else if(t.isUnsignedInteger(8))
        return "uint8_t";
    else if(t.isUnsignedInteger(32))
        return "uint32_t";
    else if(t.isUnsignedInteger(64))
        return "uint64_t";
    else if(t.isSignlessInteger(1))
        return "bool";
    else if(t.isIndex())
        return "size_t";
    else if(auto matTy = t.dyn_cast<daphne::MatrixType>())
        if(generalizeToStructure)
            return "Structure";
        else
            return "DenseMatrix_" + mlirTypeToCppTypeName(matTy.getElementType(), false);
    else if(t.isa<daphne::FrameType>())
        if(generalizeToStructure)
            return "Structure";
        else
            return "Frame";
    else if(t.isa<daphne::StringType>())
        // This becomes "const char *" (which makes perfect sense for
        // strings) when inserted into the typical "const DT *" template of
        // kernel input parameters.
        return "char";
    else if(t.isa<daphne::DaphneContextType>())
        return "DaphneContext";
    else if(auto handleTy = t.dyn_cast<daphne::HandleType>())
        return "Handle_" + mlirTypeToCppTypeName(handleTy.getDataType(), generalizeToStructure);
    throw std::runtime_error(
        "no C++ type name known for the given MLIR type"
    );
}

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
        Location loc = op->getLoc();
        if(auto strAttr = op.value().dyn_cast<StringAttr>()) {
            StringRef sr = strAttr.getValue();
#if 1
            // MLIR does not have direct support for strings. Thus, if this is
            // a string constant, we create an array large enough to store the
            // string (including a trailing null character). Then, we store all
            // characters of the string constant to that array one by one. The
            // SSA value of the constant is replaced by a pointer to i8
            // pointing to the allocated buffer.
            Type i8PtrType = LLVM::LLVMPointerType::get(
                    IntegerType::get(rewriter.getContext(), 8)
            );
            const size_t numChars = sr.size() + 1; // +1 for trailing '\0'
            const std::string str = sr.str();
            const char * chars = str.c_str();
            auto allocaOp = rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
                    op.getOperation(),
                    i8PtrType,
                    rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(numChars)),
                    1
            );
            for(size_t i = 0; i < numChars; i++) {
                std::vector<Value> indices = {
                    rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))
                };
                rewriter.create<LLVM::StoreOp>(
                        loc,
                        rewriter.create<ConstantOp>(
                                loc, rewriter.getI8IntegerAttr(chars[i])
                        ),
                        rewriter.create<LLVM::GEPOp>(
                                op->getLoc(), i8PtrType, allocaOp, indices
                        )
                );
            }
#else
            // Alternatively, we could create a global string, which would
            // yield a poiner to i8, too. However, we would need to choose a
            // unique name.
            rewriter.replaceOp(
                    op.getOperation(),
                    LLVM::createGlobalString(
                            loc, rewriter, "someName", sr,
                            LLVM::Linkage::Private // TODO Does that make sense?
                    )
            );
#endif
        }
        else {
            // Constants of all other types are lowered to an mlir::ConstantOp.
            // Note that this is a different op than mlir::daphne::ConstantOp!

            // NOTE: this fixes printing due to an error in the LLVMDialect, but is the wrong behaviour.
            //  Use this for debugging only
            /*if (auto iTy = op.getType().dyn_cast<IntegerType>()) {
                auto ty = IntegerType::get(getContext(), iTy.getWidth());
                rewriter.replaceOpWithNewOp<ConstantOp>(op.getOperation(),
                    ty,
                    IntegerAttr::get(ty, op.value().cast<IntegerAttr>().getValue()));
            }
            else {
                rewriter.replaceOpWithNewOp<ConstantOp>(op.getOperation(), op.value());
            }*/
            rewriter.replaceOpWithNewOp<ConstantOp>(op.getOperation(), op.value());
        }

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
            if(inputOutputTypes[i].dyn_cast<LLVM::LLVMPointerType>().getElementType().isa<LLVM::LLVMPointerType>()) {
                auto elType = inputOutputTypes[i].dyn_cast<LLVM::LLVMPointerType>().getElementType();
                rewriter.create<LLVM::StoreOp>(
                    loc,
                    rewriter.create<LLVM::NullOp>(loc, elType),
                    allocaOp
                );
            }
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
        Type contType = op.res().getType().dyn_cast<daphne::VariadicPackType>().getContainedType();
        Type convType = typeConverter->convertType(contType);
        rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
                op.getOperation(),
                LLVM::LLVMPointerType::get(convType),
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
        mlir::Location loc = op->getLoc();
        mlir::Value pack = operands[0];
        mlir::Value item = operands[1];
        std::vector<Value> indices = {
            rewriter.create<ConstantOp>(loc, op.posAttr())
        };
        auto addr = rewriter.create<LLVM::GEPOp>(
                loc, pack.getType(), pack, indices
        );
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
                op.getOperation(), item, addr
        );
        return success();
    }
};

class VectorizedPipelineOpLowering : public OpConversionPattern<daphne::VectorizedPipelineOp>
{
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::VectorizedPipelineOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        if (op.ctx() == nullptr) {
            op->emitOpError() << "`DaphneContext` not known";
            return failure();
        }
        auto loc = op->getLoc();
        auto numDataOperands = operands.size() - 1;// pass daphneContext separately
        LLVM::LLVMFuncOp fOp;
        {
            OpBuilder::InsertionGuard ig(rewriter);
            auto moduleOp = op->getParentOfType<ModuleOp>();
            auto &moduleBody = moduleOp.body().front();
            rewriter.setInsertionPointToStart(&moduleBody);

            static auto ix = 0;
            std::string funcName = "_vect" + std::to_string(++ix);

            auto &bodyBlock = op.body().front();

            // TODO: multi-input multi-return support
            auto i1Ty = IntegerType::get(getContext(), 1);
            auto ptrI1Ty = LLVM::LLVMPointerType::get(i1Ty);
            auto ptrPtrI1Ty = LLVM::LLVMPointerType::get(ptrI1Ty);
            auto pppI1Ty = LLVM::LLVMPointerType::get(ptrPtrI1Ty);
            // TODO: pass daphne context to function
            auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(rewriter.getContext()),
                {/*outputs...*/pppI1Ty, /*inputs...*/ptrPtrI1Ty, /*daphneContext...*/ptrI1Ty});

            fOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
            fOp.body().takeBody(op.body());
            auto &funcBlock = fOp.body().front();

            auto returnRef = funcBlock.addArgument(pppI1Ty);
            auto inputsArg = funcBlock.addArgument(ptrPtrI1Ty);
            auto daphneContext = funcBlock.addArgument(ptrI1Ty);
            // TODO: we should not create a new daphneContext, instead pass the one created in the main function
            for (auto callKernelOp : funcBlock.getOps<daphne::CallKernelOp>()) {
                callKernelOp.setOperand(callKernelOp.getNumOperands() - 1, daphneContext);
            }

            // Extract inputs from array containing them and remove the block arguments matching the old inputs of the
            // `VectorizedPipelineOp`
            rewriter.setInsertionPointToStart(&funcBlock);

            for(auto i = 0u; i < numDataOperands; ++i) {
                auto addr = rewriter.create<LLVM::GEPOp>(loc,
                    ptrPtrI1Ty,
                    inputsArg,
                    ArrayRef<Value>({
                        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))}));
                // TODO: cast for scalars etc.
                funcBlock.getArgument(0).replaceAllUsesWith(rewriter.create<LLVM::LoadOp>(loc, addr));
                funcBlock.eraseArgument(0);
            }

            // Update function block to write return value by reference instead
            auto oldReturn = funcBlock.getTerminator();
            rewriter.setInsertionPoint(oldReturn);
            for (auto i = 0u; i < oldReturn->getNumOperands(); ++i) {
                auto retVal = oldReturn->getOperand(i);
                // TODO: check how the GEPOp works exactly, and if this can be written better
                auto addr1 =
                    rewriter.create<LLVM::GEPOp>(op->getLoc(),
                        pppI1Ty,
                        returnRef,
                        ArrayRef<Value>({rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))}));
                auto addr2 = rewriter.create<LLVM::LoadOp>(op->getLoc(), addr1);
                rewriter.create<LLVM::StoreOp>(loc, retVal, addr2);
            }
            rewriter.create<ReturnOp>(loc);
            oldReturn->erase();
        }

        auto fnPtr = rewriter.create<LLVM::AddressOfOp>(loc, fOp);

        std::stringstream callee;
        callee << '_' << op->getName().stripDialect().str();

        // TODO: multi-return, pass returns as operand instead
        // Append names of result types to the kernel name.
        Operation::result_type_range resultTypes = op->getResultTypes();
        for(size_t i = 0; i < resultTypes.size(); i++)
            callee << "__" << mlirTypeToCppTypeName(resultTypes[i], false);

        std::vector<Value> newOperands;
        // TODO: support different operand types
        auto operandType = daphne::MatrixType::get(getContext(), rewriter.getF64Type());
        callee << "__" << mlirTypeToCppTypeName(operandType, false);

        // Variadic operand.
        callee << "_variadic__size_t";
        auto cvpOp = rewriter.create<daphne::CreateVariadicPackOp>(loc,
            daphne::VariadicPackType::get(rewriter.getContext(), operandType),
            rewriter.getIndexAttr(numDataOperands));
        for(size_t k = 0; k < numDataOperands; k++) {
            rewriter.create<daphne::StoreVariadicPackOp>(loc, cvpOp, operands[k], rewriter.getIndexAttr(k));
        }
        newOperands.push_back(cvpOp);
        newOperands.push_back(rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexAttr(numDataOperands)));

        // Add array of split enums
        callee << "__int64_t";
        std::vector<Value> splitConsts;
        for(auto split : op.splits()) {
            splitConsts.push_back(rewriter.create<ConstantOp>(loc, split));
        }
        newOperands.push_back(convertToArray(loc, rewriter, rewriter.getI64Type(), splitConsts));

        // Add array of combine enums
        callee << "__int64_t";
        std::vector<Value> combineConsts;
        for(auto combine : op.combines()) {
            combineConsts.push_back(rewriter.create<ConstantOp>(loc, combine));
        }
        newOperands.push_back(convertToArray(loc, rewriter, rewriter.getI64Type(), combineConsts));

        // TODO: pass function pointer with special placeholder instead of `void`
        callee << "__void";
        newOperands.push_back(fnPtr);
        // Add ctx
        newOperands.push_back(operands.back());

        // Create a CallKernelOp for the kernel function to call and return
        // success().
        auto kernel = rewriter.create<daphne::CallKernelOp>(
            loc,
            callee.str(),
            newOperands,
            op->getResultTypes()
        );
        rewriter.replaceOp(op, kernel.getResults());
        return success();
    }
private:
    static Value convertToArray(Location loc, ConversionPatternRewriter &rewriter, Type valueTy, ValueRange values)
    {
        auto valuePtrTy = LLVM::LLVMPointerType::get(valueTy);
        auto array = rewriter.create<LLVM::AllocaOp>(loc,
            valuePtrTy,
            Value(rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(values.size()))));
        for(auto i = 0u; i < values.size(); ++i) {
            Value cstI = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
            auto addr = rewriter.create<LLVM::GEPOp>(loc, valuePtrTy, array, ArrayRef<Value>({cstI}));
            rewriter.create<LLVM::StoreOp>(loc, values[i], addr);
        }
        return array;
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

    LowerToLLVMOptions llvmOptions(&getContext());
    llvmOptions.emitCWrappers = true;
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);
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
    typeConverter.addConversion([&](daphne::StringType t)
    {
        return LLVM::LLVMPointerType::get(
                IntegerType::get(t.getContext(), 8));
    });
    typeConverter.addConversion([&](daphne::VariadicPackType t)
    {
        return LLVM::LLVMPointerType::get(
                typeConverter.convertType(t.getContainedType())
        );
    });
    typeConverter.addConversion([&](daphne::DaphneContextType t)
    {
        return LLVM::LLVMPointerType::get(
                IntegerType::get(t.getContext(), 1));
    });
    typeConverter.addConversion([&](daphne::HandleType t)
    {
      return LLVM::LLVMPointerType::get(
          IntegerType::get(t.getContext(), 1));
    });
    typeConverter.addConversion([&](daphne::FileType t)
    {
      return LLVM::LLVMPointerType::get(
          IntegerType::get(t.getContext(), 1));
    });
    typeConverter.addConversion([&](daphne::DescriptorType t)
    {
      return LLVM::LLVMPointerType::get(
          IntegerType::get(t.getContext(), 1));
    });
    typeConverter.addConversion([&](daphne::TargetType t)
    {
      return LLVM::LLVMPointerType::get(
          IntegerType::get(t.getContext(), 1));
    });

    LLVMConversionTarget target(getContext());

    // populate dialect conversions
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    target.addLegalOp<ModuleOp>();

    patterns.insert<
            CallKernelOpLowering,
            CreateVariadicPackOpLowering,
            VectorizedPipelineOpLowering
    >(typeConverter, &getContext());
    patterns.insert<
            ConstantOpLowering,
            ReturnOpLowering,
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
