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
#include "compiler/CompilerUtils.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include <memory>
#include <utility>
#include <vector>
#include <iostream>

using namespace mlir;

// Optional attribute of CallKernelOp, which indicates that all results shall
// be combined into a single variadic result.
const std::string ATTR_HASVARIADICRESULTS = "hasVariadicResults";

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

struct CastOpLowering : public OpRewritePattern<daphne::CastOp> {
    using OpRewritePattern<daphne::CastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::CastOp op,
                                  PatternRewriter &rewriter) const final {
        if(op.isTrivialCast() || op.isMatrixPropertyCast()) {
            rewriter.replaceOp(op, op.getOperand());
            return success();
        }
        return failure();
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
#if 1
            rewriter.replaceOpWithNewOp<ConstantOp>(op.getOperation(), op.value());
#else
            // NOTE: this fixes printing due to an error in the LLVMDialect, but is the wrong behaviour.
            //  Use this for debugging only
            if (auto iTy = op.getType().dyn_cast<IntegerType>()) {
                auto ty = IntegerType::get(getContext(), iTy.getWidth());
                rewriter.replaceOpWithNewOp<ConstantOp>(op.getOperation(),
                    ty,
                    IntegerAttr::get(ty, op.value().cast<IntegerAttr>().getValue()));
            }
            else {
                rewriter.replaceOpWithNewOp<ConstantOp>(op.getOperation(), op.value());
            }
#endif
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
                                                     TypeRange operandTypes,
                                                     bool hasVarRes,
                                                     Type indexType)
    {
        llvm::SmallVector<Type, 5> args;
        
        // --------------------------------------------------------------------
        // Results
        // --------------------------------------------------------------------
        
        const size_t numRes = resultTypes.size();
        if(hasVarRes) { // combine all results into one variadic result
            // TODO Support individual result types, at least if they are all
            // mapped to the superclass Structure (see #397).
            // Check if all results have the same type.
            Type t0 = resultTypes[0];
            Type mt0 = t0.dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr();
            for(size_t i = 1; i < numRes; i++)
                if(mt0 != resultTypes[i].dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr())
                    throw std::runtime_error(
                            "all results of a CallKernelOp must have the same "
                            "type to combine them into a single variadic result"
                    );
            // Wrap the common result type into a pointer, since we need an
            // array of that type.
            args.push_back(LLVM::LLVMPointerType::get(
                    typeConverter->isLegal(t0)
                    ? t0
                    : typeConverter->convertType(t0)
            ));
        }
        else // typical case
            for (auto type : resultTypes) {
                if (typeConverter->isLegal(type)) {
                    args.push_back(type);
                }
                else if (failed(typeConverter->convertType(type, args)))
                    emitError(loc) << "Couldn't convert result type `" << type << "`\n";
            }
        
        // --------------------------------------------------------------------
        // Operands
        // --------------------------------------------------------------------
        
        if(hasVarRes)
            // Create a parameter for passing the number of results in the
            // single variadic result.
            args.push_back(typeConverter->isLegal(indexType)
                    ? indexType
                    : typeConverter->convertType(indexType));
        
        for (auto type : operandTypes) {
            if (typeConverter->isLegal(type)) {
                args.push_back(type);
            }
            else if (failed(typeConverter->convertType(type, args)))
                emitError(loc) << "Couldn't convert operand type `" << type << "`\n";
        }

        // --------------------------------------------------------------------
        // Create final LLVM types
        // --------------------------------------------------------------------
        
        std::vector<Type> argsLLVM;
        for (size_t i = 0; i < args.size(); i++) {
            Type type = args[i];
            // Wrap result types into a pointer, since we want to pass results
            // "by-reference". Note that a variadic result is not passed
            // "by-reference", since we don't need to change the array itself
            // in a kernel.
            if (!hasVarRes && i < numRes) {
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
        // Whether all results of the operation shall be combined into one
        // vardiadic result. If this is false (typical case), we pass a
        // separate nullptr for each result to the kernel. If it is true, we
        // create an array with the number of results, fill it with nullptrs,
        // and pass that to the kernel (variadic results).
        const bool hasVarRes = op->hasAttr(ATTR_HASVARIADICRESULTS)
                ? op->getAttr(ATTR_HASVARIADICRESULTS).dyn_cast<BoolAttr>().getValue()
                : false;
        
        auto module = op->getParentOfType<ModuleOp>();
        auto loc = op.getLoc();

        auto inputOutputTypes = getLLVMInputOutputTypes(
                                                        loc, rewriter.getContext(), typeConverter,
                                                        op.getResultTypes(), ValueRange(operands).getTypes(),
                                                        hasVarRes, rewriter.getIndexType());

        // create function protoype and get `FlatSymbolRefAttr` to it
        auto kernelRef = getOrInsertFunctionAttr(
                                                 rewriter, module, op.getCalleeAttr().getValue(),
                                                 getKernelFuncSignature(rewriter.getContext(), inputOutputTypes));

        auto kernelOperands = allocOutputReferences(loc, rewriter, operands, inputOutputTypes, op->getNumResults(), hasVarRes);

        // call function
        rewriter.create<CallOp>(
                loc, kernelRef,
                /*no return value*/ LLVM::LLVMVoidType::get(rewriter.getContext()),
                kernelOperands);
        rewriter.replaceOp(op, dereferenceOutputs(loc, rewriter, module,
                                                  op->getNumResults(),
                                                  hasVarRes, kernelOperands));
        return success();
    }

private:

    static std::vector<Value>
    dereferenceOutputs(Location &loc, PatternRewriter &rewriter, ModuleOp &module,
                       size_t numResults, bool hasVarRes, std::vector<Value> kernelOperands)
    {
        // transformed results
        std::vector<Value> results;
        
        if(hasVarRes) { // combine all results into one variadic result
            for(size_t i = 0; i < numResults; i++) {
                std::vector<Value> indices = {rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))};
                results.push_back(rewriter.create<LLVM::LoadOp>(
                        loc,
                        rewriter.create<LLVM::GEPOp>(
                                loc,
                                kernelOperands[0].getType(),
                                kernelOperands[0],
                                indices
                        )
                ));
            }
        }
        else // typical case
            for (size_t i = 0; i < numResults; i++) {
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
                          std::vector<Type> inputOutputTypes, size_t numRes, bool hasVarRes) const
    {

        std::vector<Value> kernelOperands;
        
        // --------------------------------------------------------------------
        // Results
        // --------------------------------------------------------------------
        
        if(hasVarRes) { // combine all results into one variadic result
            // Allocate an array of numRes elements.
            auto allocaOp = rewriter.create<LLVM::AllocaOp>(
                    loc,
                    inputOutputTypes[0],
                    rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(numRes)).getResult()
            );
            kernelOperands.push_back(allocaOp);

            // If the type of this result parameter is a pointer (i.e. when it
            // represents a matrix or frame), then initialize the allocated
            // element with a null pointer (required by the kernels). Otherwise
            // (i.e. when it represents a scalar), initialization is not
            // required.
            Type elType = inputOutputTypes[0].dyn_cast<LLVM::LLVMPointerType>().getElementType();
            if(elType.isa<LLVM::LLVMPointerType>()) {
                for(size_t i = 0; i < numRes; i++) {
                    std::vector<Value> indices = {rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))};
                    rewriter.create<LLVM::StoreOp>(
                        loc,
                        rewriter.create<LLVM::NullOp>(loc, elType),
                        rewriter.create<LLVM::GEPOp>(
                                loc, inputOutputTypes[0], allocaOp, indices
                        )
                    );
                }
            }
        }
        else { // typical case
            // Constant of 1 for AllocaOp of output.
            Value cst1 = rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
            
            for (size_t i = 0; i < numRes; i++) {
                // Allocate space for a single element.
                auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, inputOutputTypes[i], cst1);
                kernelOperands.push_back(allocaOp);

                // If the type of this result parameter is a pointer (i.e. when it
                // represents a matrix or frame), then initialize the allocated
                // element with a null pointer (required by the kernels). Otherwise
                // (i.e. when it represents a scalar), initialization is not
                // required.
                Type elType = inputOutputTypes[i].dyn_cast<LLVM::LLVMPointerType>().getElementType();
                if(elType.isa<LLVM::LLVMPointerType>()) {
                    rewriter.create<LLVM::StoreOp>(
                        loc,
                        rewriter.create<LLVM::NullOp>(loc, elType),
                        allocaOp
                    );
                }
            }
        }
        
        // --------------------------------------------------------------------
        // Operands
        // --------------------------------------------------------------------
        
        if(hasVarRes)
            // Insert the number of results in the variadic result as a constant.
            kernelOperands.push_back(rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(numRes)));
        
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
        auto elementType = pack.getType().cast<LLVM::LLVMPointerType>().getElementType();
        std::vector<Value> indices = {
            rewriter.create<ConstantOp>(loc, op.posAttr())
        };
        auto addr = rewriter.create<LLVM::GEPOp>(
                loc, pack.getType(), pack, indices
        );
        Type itemType = item.getType();
        if (itemType != elementType) {
            if (elementType.isa<LLVM::LLVMPointerType>()) {
                if(itemType.isSignedInteger())
                    item = rewriter.create<LLVM::SExtOp>(loc, rewriter.getI64Type(), item);
                else if(itemType.isUnsignedInteger() || itemType.isSignlessInteger())
                    item = rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI64Type(), item);
                else if(itemType.isa<FloatType>()) {
                    item = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), item);
                    item = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), item);
                }
                else
                    throw std::runtime_error("itemType is an unsupported type");
                item = rewriter.create<LLVM::IntToPtrOp>(loc, elementType, item);
            }
            else
                throw std::runtime_error(
                        "casting to a non-pointer type in "
                        "StoreVariadicPackOpLowering is not implemented yet"
                );
        }
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
                op.getOperation(), item, addr
        );
        return success();
    }
};

class VectorizedPipelineOpLowering : public OpConversionPattern<daphne::VectorizedPipelineOp>
{
    const DaphneUserConfig& cfg;

public:
    explicit VectorizedPipelineOpLowering(TypeConverter &typeConverter, MLIRContext *context, const DaphneUserConfig &cfg)
            : OpConversionPattern(typeConverter, context), cfg(cfg) {}

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
        auto numDataOperands = op.inputs().size();
        std::vector<mlir::Value> func_ptrs;

        auto i1Ty = IntegerType::get(getContext(), 1);
        auto ptrI1Ty = LLVM::LLVMPointerType::get(i1Ty);
        auto ptrPtrI1Ty = LLVM::LLVMPointerType::get(ptrI1Ty);
        auto pppI1Ty = LLVM::LLVMPointerType::get(ptrPtrI1Ty);

        LLVM::LLVMFuncOp fOp;
        {
            OpBuilder::InsertionGuard ig(rewriter);
            auto moduleOp = op->getParentOfType<ModuleOp>();
            auto &moduleBody = moduleOp.body().front();
            rewriter.setInsertionPointToStart(&moduleBody);

            static auto ix = 0;
            std::string funcName = "_vect" + std::to_string(++ix);

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
                Value val = rewriter.create<LLVM::LoadOp>(loc, addr);
                auto expTy = typeConverter->convertType(op.inputs().getType()[i]);
                if (expTy != val.getType()) {
                    // casting for scalars
                    val = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), val);
                    if(expTy.isa<IntegerType>())
                        val = rewriter.create<LLVM::TruncOp>(loc, expTy, val);
                    else if(expTy.isa<FloatType>()) {
                        val = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF64Type(), val);
                        val = rewriter.create<LLVM::FPTruncOp>(loc, expTy, val);
                    }
                    else
                        throw std::runtime_error("expTy is an unsupported type");
                }
                funcBlock.getArgument(0).replaceAllUsesWith(val);
                funcBlock.eraseArgument(0);
            }

            // Update function block to write return value by reference instead
            auto oldReturn = funcBlock.getTerminator();
            rewriter.setInsertionPoint(oldReturn);
            for (auto i = 0u; i < oldReturn->getNumOperands(); ++i) {
                auto retVal = oldReturn->getOperand(i);
                // TODO: check how the GEPOp works exactly, and if this can be written better
                auto addr1 = rewriter.create<LLVM::GEPOp>(op->getLoc(), pppI1Ty, returnRef, ArrayRef<Value>(
                        {rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))}));
                auto addr2 = rewriter.create<LLVM::LoadOp>(op->getLoc(), addr1);
                rewriter.create<LLVM::StoreOp>(loc, retVal, addr2);
            }
            // Replace the old ReturnOp with operands by a new ReturnOp without
            // operands.
            rewriter.replaceOpWithNewOp<ReturnOp>(oldReturn);
        }

        auto fnPtr = rewriter.create<LLVM::AddressOfOp>(loc, fOp);

        func_ptrs.push_back(fnPtr);

        if(cfg.use_cuda && !op.cuda().getBlocks().empty()) {
            LLVM::LLVMFuncOp fOp2;
            {
                OpBuilder::InsertionGuard ig(rewriter);
                auto moduleOp = op->getParentOfType<ModuleOp>();
                auto &moduleBody = moduleOp.body().front();
                rewriter.setInsertionPointToStart(&moduleBody);

                static auto ix = 0;
                std::string funcName = "_vect_cuda" + std::to_string(++ix);

                auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(rewriter.getContext()),
            {/*outputs...*/pppI1Ty, /*inputs...*/ ptrPtrI1Ty, /*daphneContext...*/ptrI1Ty});

                fOp2 = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
                fOp2.body().takeBody(op.cuda());
                auto &funcBlock = fOp2.body().front();

                auto returnRef = funcBlock.addArgument(pppI1Ty);
                auto inputsArg = funcBlock.addArgument(ptrPtrI1Ty);
                auto daphneContext = funcBlock.addArgument(ptrI1Ty);

                // TODO: we should not create a new daphneContext, instead pass the one created in the main function
                for (auto callKernelOp: funcBlock.getOps<daphne::CallKernelOp>()) {
                    callKernelOp.setOperand(callKernelOp.getNumOperands() - 1, daphneContext);
                }

                // Extract inputs from array containing them and remove the block arguments matching the old inputs of the
                // `VectorizedPipelineOp`
                rewriter.setInsertionPointToStart(&funcBlock);

                for (auto i = 0u; i < numDataOperands; ++i) {
                    auto addr = rewriter.create<LLVM::GEPOp>(loc, ptrPtrI1Ty, inputsArg, ArrayRef<Value>({
                            rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))}));
                    Value val = rewriter.create<LLVM::LoadOp>(loc, addr);
                    auto expTy = typeConverter->convertType(op.inputs().getType()[i]);
                    if (expTy != val.getType()) {
                        // casting for scalars
                        val = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), val);
                        val = rewriter.create<LLVM::BitcastOp>(loc, expTy, val);
                    }
                    funcBlock.getArgument(0).replaceAllUsesWith(val);
                    funcBlock.eraseArgument(0);
                }

                // Update function block to write return value by reference instead
                auto oldReturn = funcBlock.getTerminator();
                rewriter.setInsertionPoint(oldReturn);
                for (auto i = 0u; i < oldReturn->getNumOperands(); ++i) {
                    auto retVal = oldReturn->getOperand(i);
                    // TODO: check how the GEPOp works exactly, and if this can be written better
                    auto addr1 = rewriter.create<LLVM::GEPOp>(op->getLoc(), pppI1Ty, returnRef, ArrayRef<Value>(
                        {rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i))}));
                    auto addr2 = rewriter.create<LLVM::LoadOp>(op->getLoc(), addr1);
                    rewriter.create<LLVM::StoreOp>(loc, retVal, addr2);
                }
                // Replace the old ReturnOp with operands by a new ReturnOp without
                // operands.
                rewriter.replaceOpWithNewOp<ReturnOp>(oldReturn);
            }

            auto fnPtr2 = rewriter.create<LLVM::AddressOfOp>(loc, fOp2);

            func_ptrs.push_back(fnPtr2);
        }
        std::stringstream callee;
        callee << '_' << op->getName().stripDialect().str();

        // Get some information on the results.
        Operation::result_type_range resultTypes = op->getResultTypes();
        const size_t numRes = op->getNumResults();
        
        if(numRes > 0) {
            // TODO Support individual types for all outputs (see #397).
            // Check if all results have the same type.
            Type mt0 = resultTypes[0].dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr();
            for(size_t i = 1; i < numRes; i++)
                if(mt0 != resultTypes[i].dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr())
                    throw std::runtime_error(
                            "encountered a vectorized pipelines with different "
                            "result types, but at the moment we require all "
                            "results to have the same type"
                    );
            
            // Append the name of the common type of all results to the kernel name.
            callee << "__" << CompilerUtils::mlirTypeToCppTypeName(resultTypes[0]) << "_variadic__size_t";
        }

        mlir::Type operandType;
        std::vector<Value> newOperands;
        if(numRes > 0) {
            auto m32type = rewriter.getF32Type();
            auto m64type = rewriter.getF64Type();
            auto res_elem_type = op->getResult(0).getType().dyn_cast<mlir::daphne::MatrixType>().getElementType();
            if(res_elem_type == m64type)
                operandType = daphne::MatrixType::get(getContext(), m64type);
            else if(res_elem_type == m32type)
                operandType = daphne::MatrixType::get(getContext(), m32type);
            else {
                std::string str;
                llvm::raw_string_ostream output(str);
                op->getResult(0).getType().print(output);
                throw std::runtime_error("Unsupported result type for vectorizedPipeline op: " + str);
            }
        }
        else {
            throw std::runtime_error("vectorizedPipelineOp without outputs not supported at the moment!");
        }

        // Handle variadic operands isScalar and inputs (both share numInputs).
        auto idxAttrNumInputs = rewriter.getIndexAttr(numDataOperands);
        // For isScalar.
        callee << "__bool";
        auto vpScalar = rewriter.create<daphne::CreateVariadicPackOp>(loc,
            daphne::VariadicPackType::get(rewriter.getContext(), rewriter.getI1Type()),
            idxAttrNumInputs);
        // For inputs and numInputs.
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(operandType, true);
        callee << "_variadic__size_t";
        auto vpInputs = rewriter.create<daphne::CreateVariadicPackOp>(loc,
            daphne::VariadicPackType::get(rewriter.getContext(), operandType),
            idxAttrNumInputs);
        // Populate the variadic packs for isScalar and inputs.
        for(size_t k = 0; k < numDataOperands; k++) {
            auto idxAttrK = rewriter.getIndexAttr(k);
            rewriter.create<daphne::StoreVariadicPackOp>(
                    loc,
                    vpScalar,
                    rewriter.create<daphne::ConstantOp>(
                            loc,
                            // We assume this input to be a scalar if its type
                            // has not been converted to a pointer type.
                            !operands[k].getType().isa<LLVM::LLVMPointerType>()
                    ),
                    idxAttrK
            );
            rewriter.create<daphne::StoreVariadicPackOp>(
                    loc, vpInputs, operands[k], idxAttrK
            );
        }
        newOperands.push_back(vpScalar);
        newOperands.push_back(vpInputs);
        newOperands.push_back(rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexAttr(numDataOperands)));

        auto numOutputs = op.getNumResults();
        // Variadic num rows operands.
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(rewriter.getIntegerType(64, true));
        auto rowsOperands = operands.drop_front(numDataOperands);
        newOperands
            .push_back(convertToArray(loc, rewriter, rewriter.getI64Type(), rowsOperands.take_front(numOutputs)));
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(rewriter.getIntegerType(64, true));
        auto colsOperands = rowsOperands.drop_front(numOutputs);
        newOperands.push_back(convertToArray(loc, rewriter, rewriter.getI64Type(), colsOperands.take_front(numOutputs)));

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

        callee << "__size_t";
        newOperands.push_back(rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexAttr(func_ptrs.size())));
        callee << "__void_variadic";
        newOperands.push_back(convertToArray(loc, rewriter, ptrPtrI1Ty, func_ptrs));
//        newOperands.push_back(fnPtr);

        // Add ctx
//        newOperands.push_back(operands.back());
        if (op.ctx() == nullptr) {
            op->emitOpError() << "`DaphneContext` not known";
            return failure();
        }
        else
            newOperands.push_back(op.ctx());
        // Create a CallKernelOp for the kernel function to call and return
        // success().
        auto kernel = rewriter.create<daphne::CallKernelOp>(
            loc,
            callee.str(),
            newOperands,
            resultTypes
        );
        kernel->setAttr(ATTR_HASVARIADICRESULTS, rewriter.getBoolAttr(true));
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
            auto val = values[i];
            if (val.getType() != valueTy) {
                val = rewriter.create<LLVM::BitcastOp>(loc, valueTy, val);
            }
            rewriter.create<LLVM::StoreOp>(loc, val, addr);
        }
        return array;
    }
};

class GenericCallOpLowering : public OpConversionPattern<daphne::GenericCallOp>
{
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(daphne::GenericCallOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<CallOp>(op, op.callee(), op->getResultTypes(), operands);
        return success();
    }
};

namespace
{
    struct DaphneLowerToLLVMPass
    : public PassWrapper<DaphneLowerToLLVMPass, OperationPass<ModuleOp>>
    {
		explicit DaphneLowerToLLVMPass(const DaphneUserConfig& cfg) : cfg(cfg) { }
		const DaphneUserConfig& cfg;

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
    // TODO: just create CWrappers for `main` and UDFs (currently vectorized pipelines are also emitted)
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

    // for trivial casts no lowering to kernels -> higher benefit
    patterns.insert<CastOpLowering>(&getContext(), 2);
    patterns.insert<
            CallKernelOpLowering,
            CreateVariadicPackOpLowering>(typeConverter, &getContext());

    patterns.insert<VectorizedPipelineOpLowering>(typeConverter, &getContext(), cfg);

    patterns.insert<
            ConstantOpLowering,
            ReturnOpLowering,
            StoreVariadicPackOpLowering,
            GenericCallOpLowering
    >(&getContext());

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createLowerToLLVMPass(const DaphneUserConfig& cfg)
{
    return std::make_unique<DaphneLowerToLLVMPass>(cfg);
}
