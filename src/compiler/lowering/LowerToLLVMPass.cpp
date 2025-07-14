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

#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include <util/ErrorHandler.h>

#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/Analysis/SliceAnalysis.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace mlir;

// Remark on the creation of mlir::LLVM::AllocaOp
// ==============================================
// This pass creates an mlir::LLVM::AllocaOp in several places and for various
// purposes, e.g., to store the result pointer of a kernel call, for variadic
// operands/results, etc. AllocaOp should not be inside a loop, as its repeated
// execution at run-time can lead to a stack overflow (depending on the number
// of iterations, the number of AllocaOps inside the loop, and the stack size).
// The reason is that the memory allocated by AllocaOp is freed only at the end
// of the scope (i.e., function). To avoid such problems, we don't create
// AllocaOps at the original insertion point of the rewriter, but at the
// beginning of function surrounding the currently considered op. To this end,
// we use the rewriter's ability to switch between different insertion points.
// Note that the memory allocated by an AllocaOp can be reused by multiple
// repeated kernel calls.

// Optional attribute of CallKernelOp, which indicates that all results shall
// be combined into a single variadic result.
const std::string ATTR_HASVARIADICRESULTS = "hasVariadicResults";

struct ReturnOpLowering : public OpRewritePattern<daphne::ReturnOp> {
    using OpRewritePattern<daphne::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::ReturnOp op, PatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
        return success();
    }
};

struct CastOpLowering : public OpRewritePattern<daphne::CastOp> {
    using OpRewritePattern<daphne::CastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::CastOp op, PatternRewriter &rewriter) const final {
        if (op.isTrivialCast() || op.isRemovePropertyCast()) {
            rewriter.replaceOp(op, op.getOperand());
            return success();
        }
        return failure();
    }
};

/// ConstantOp lowering for types not handled before (str)

class ConstantOpLowering : public OpConversionPattern<daphne::ConstantOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::ConstantOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        if (auto strAttr = op.getValue().dyn_cast<StringAttr>()) {
            StringRef sr = strAttr.getValue();
#if 1
            // MLIR does not have direct support for strings. Thus, if this is
            // a string constant, we create an array large enough to store the
            // string (including a trailing null character). Then, we store all
            // characters of the string constant to that array one by one. The
            // SSA value of the constant is replaced by a pointer to i8
            // pointing to the allocated buffer.
            Type i8PtrType = LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
            const size_t numChars = sr.size() + 1; // +1 for trailing '\0'
            const std::string str = sr.str();
            const char *chars = str.c_str();

            // We could assume that the daphne::ConstantOp `op` is *not* inside
            // a loop, because constants are typically moved to the top of a
            // function during canonicalization. Consequently, we would not need
            // to change the insertion point. However, being defensive, we still
            // do it.

            // Set the insertion point to the beginning of the function
            // surrounding this ConstantOp (see comment on AllocaOp above).
            OpBuilder::InsertPoint ipHere = rewriter.saveInsertionPoint();
            Block &fb = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>().getBody().front();
            rewriter.setInsertionPointToStart(&fb);

            auto allocaOp = rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
                op.getOperation(), i8PtrType,
                rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(numChars)), 1);

            // Go back to the original insertion point.
            rewriter.restoreInsertionPoint(ipHere);

            for (size_t i = 0; i < numChars; i++) {
                std::vector<Value> indices = {rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i))};
                rewriter.create<LLVM::StoreOp>(
                    loc, rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(chars[i])),
                    rewriter.create<LLVM::GEPOp>(op->getLoc(), i8PtrType, allocaOp, indices));
            }
#else
            // Alternatively, we could create a global string, which would
            // yield a poiner to i8, too. However, we would need to choose a
            // unique name.
            rewriter.replaceOp(op.getOperation(),
                               LLVM::createGlobalString(loc, rewriter, "someName", sr,
                                                        LLVM::Linkage::Private // TODO Does that make sense?
                                                        ));
#endif
        } else {
            // Constants of all other types are lowered to an
            // mlir::arith::ConstantOp. Note that this is a different op than
            // mlir::daphne::ConstantOp!
#if 1
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(op.getOperation(), op.getValue());
#else
            // NOTE: this fixes printing due to an error in the LLVMDialect, but
            // is the wrong behaviour.
            //  Use this for debugging only
            if (auto iTy = op.getType().dyn_cast<IntegerType>()) {
                auto ty = IntegerType::get(getContext(), iTy.getWidth());
                rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                    op.getOperation(), ty, IntegerAttr::get(ty, op.getValue().cast<IntegerAttr>().getValue()));
            } else {
                rewriter.replaceOpWithNewOp<arith::ConstantOp>(op.getOperation(), op.getValue());
            }
#endif
        }

        return success();
    }
};

class CallKernelOpLowering : public OpConversionPattern<daphne::CallKernelOp> {

    static std::vector<Type> getLLVMInputOutputTypes(Location &loc, MLIRContext *context, TypeConverter *typeConverter,
                                                     TypeRange resultTypes, TypeRange operandTypes, bool hasVarRes,
                                                     Type indexType) {
        llvm::SmallVector<Type, 5> args;

        // --------------------------------------------------------------------
        // Results
        // --------------------------------------------------------------------

        const size_t numRes = resultTypes.size();
        if (hasVarRes) { // combine all results into one variadic result
            // TODO Support individual result types, at least if they are all
            // mapped to the superclass Structure (see #397).
            // Check if all results have the same type.
            Type t0 = resultTypes[0];
            Type mt0 = t0.dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr();
            for (size_t i = 1; i < numRes; i++)
                if (mt0 != resultTypes[i].dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr()) {
                    throw ErrorHandler::compilerError(loc, "LowerToLLVMPass",
                                                      "all results of a CallKernelOp must have the same "
                                                      "type to combine them into a single variadic result");
                }
            // Wrap the common result type into a pointer, since we need an
            // array of that type.
            args.push_back(
                LLVM::LLVMPointerType::get(typeConverter->isLegal(t0) ? t0 : typeConverter->convertType(t0)));

        } else // typical case
            for (auto type : resultTypes) {
                if (typeConverter->isLegal(type)) {
                    args.push_back(type);
                } else if (failed(typeConverter->convertType(type, args)))
                    emitError(loc) << "Couldn't convert result type `" << type << "`\n";
            }

        // --------------------------------------------------------------------
        // Operands
        // --------------------------------------------------------------------

        if (hasVarRes)
            // Create a parameter for passing the number of results in the
            // single variadic result.
            args.push_back(typeConverter->isLegal(indexType) ? indexType : typeConverter->convertType(indexType));
        for (auto type : operandTypes) {
            if (typeConverter->isLegal(type)) {
                args.push_back(type);
            } else if (failed(typeConverter->convertType(type, args)))
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

    static FlatSymbolRefAttr getOrInsertFunctionAttr(OpBuilder &rewriter, ModuleOp module, llvm::StringRef funcName,
                                                     LLVM::LLVMFunctionType llvmFnType) {
        auto *context = module.getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
            return SymbolRefAttr::get(context, funcName);

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, llvmFnType);
        return SymbolRefAttr::get(context, funcName);
    }

    static LLVM::LLVMFunctionType getKernelFuncSignature(MLIRContext *context, std::vector<Type> argsLLVM) {
        return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), argsLLVM, false);
    }

  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::CallKernelOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
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

        auto inputOutputTypes =
            getLLVMInputOutputTypes(loc, rewriter.getContext(), typeConverter, op.getResultTypes(),
                                    ValueRange(adaptor.getOperands()).getTypes(), hasVarRes, rewriter.getIndexType());
        // create function protoype and get `FlatSymbolRefAttr` to it
        auto kernelRef = getOrInsertFunctionAttr(rewriter, module, op.getCalleeAttr().getValue(),
                                                 getKernelFuncSignature(rewriter.getContext(), inputOutputTypes));
        auto kernelOperands = allocOutputReferences(loc, rewriter, adaptor.getOperands(), inputOutputTypes,
                                                    op->getNumResults(), hasVarRes, op);
        // call function
        // The kernel call has an empty list of return types, because our
        // kernel(-wrapper)s generally return via parameters.
        TypeRange ts;
        rewriter.create<func::CallOp>(loc, kernelRef, ts, kernelOperands);
        rewriter.replaceOp(op,
                           dereferenceOutputs(loc, rewriter, module, op->getNumResults(), hasVarRes, kernelOperands));
        return success();
    }

  private:
    static std::vector<Value> dereferenceOutputs(Location &loc, PatternRewriter &rewriter, ModuleOp &module,
                                                 size_t numResults, bool hasVarRes, std::vector<Value> kernelOperands) {
        // transformed results
        std::vector<Value> results;

        if (hasVarRes) { // combine all results into one variadic result
            for (size_t i = 0; i < numResults; i++) {
                std::vector<Value> indices = {rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i))};
                results.push_back(rewriter.create<LLVM::LoadOp>(
                    loc, rewriter.create<LLVM::GEPOp>(loc, kernelOperands[0].getType(), kernelOperands[0], indices)));
            }
        } else // typical case
            for (size_t i = 0; i < numResults; i++) {
                // dereference output
                auto value = kernelOperands[i];
                // load element (dereference)
                auto resultVal = rewriter.create<LLVM::LoadOp>(loc, value);

                results.push_back(resultVal);
            }

        return results;
    }

    std::vector<Value> allocOutputReferences(Location &loc, PatternRewriter &rewriter, ValueRange operands,
                                             std::vector<Type> inputOutputTypes, size_t numRes, bool hasVarRes,
                                             daphne::CallKernelOp op) const {

        std::vector<Value> kernelOperands;

        // Obtain an insertion point at the beginning of the function
        // surrounding this CallKernelOp (see comment on AllocaOp above).
        OpBuilder::InsertPoint ipHere = rewriter.saveInsertionPoint();
        Block &fb = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>().getBody().front();
        rewriter.setInsertionPointToStart(&fb);
        OpBuilder::InsertPoint ipFuncStart = rewriter.saveInsertionPoint();
        rewriter.restoreInsertionPoint(ipHere);

        // --------------------------------------------------------------------
        // Results
        // --------------------------------------------------------------------

        if (hasVarRes) { // combine all results into one variadic result
            // Allocate an array of numRes elements.

            // Set the insertion point to the beginning of the function (see
            // comment on AllocaOp above).
            ipHere = rewriter.saveInsertionPoint();
            rewriter.restoreInsertionPoint(ipFuncStart);
            auto allocaOp = rewriter.create<LLVM::AllocaOp>(
                loc, inputOutputTypes[0],
                rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(numRes)).getResult());
            ipFuncStart = rewriter.saveInsertionPoint();

            // Go back to the original insertion point.
            rewriter.restoreInsertionPoint(ipHere);

            kernelOperands.push_back(allocaOp);

            // If the type of this result parameter is a pointer (i.e. when it
            // represents a matrix or frame), then initialize the allocated
            // element with a null pointer (required by the kernels). Otherwise
            // (i.e. when it represents a scalar), initialization is not
            // required.
            Type elType = inputOutputTypes[0].dyn_cast<LLVM::LLVMPointerType>().getElementType();
            if (llvm::isa<LLVM::LLVMPointerType>(elType)) {
                for (size_t i = 0; i < numRes; i++) {
                    std::vector<Value> indices = {
                        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i))};
                    rewriter.create<LLVM::StoreOp>(
                        loc, rewriter.create<LLVM::NullOp>(loc, elType),
                        rewriter.create<LLVM::GEPOp>(loc, inputOutputTypes[0], allocaOp, indices));
                }
            }
        } else { // typical case
            // Constant of 1 for AllocaOp of output.
            // Set the insertion point to the beginning of the function (see
            // comment on AllocaOp above).
            ipHere = rewriter.saveInsertionPoint();
            rewriter.restoreInsertionPoint(ipFuncStart);
            Value cst1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
            ipFuncStart = rewriter.saveInsertionPoint();

            // Go back to the original insertion point.
            rewriter.restoreInsertionPoint(ipHere);

            for (size_t i = 0; i < numRes; i++) {
                // Allocate space for a single element.
                // Set the insertion point to the beginning of the function (see
                // comment on AllocaOp above).
                ipHere = rewriter.saveInsertionPoint();
                rewriter.restoreInsertionPoint(ipFuncStart);
                auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, inputOutputTypes[i], cst1);
                ipFuncStart = rewriter.saveInsertionPoint();
                kernelOperands.push_back(allocaOp);

                // Go back to the original insertion point.
                rewriter.restoreInsertionPoint(ipHere);

                // If the type of this result parameter is a pointer (i.e. when
                // it represents a matrix or frame), then initialize the
                // allocated element with a null pointer (required by the
                // kernels). Otherwise (i.e. when it represents a scalar),
                // initialization is not required.
                Type elType = inputOutputTypes[i].dyn_cast<LLVM::LLVMPointerType>().getElementType();
                if (llvm::isa<LLVM::LLVMPointerType>(elType)) {
                    rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::NullOp>(loc, elType), allocaOp);
                }
            }
        }

        // --------------------------------------------------------------------
        // Operands
        // --------------------------------------------------------------------

        if (hasVarRes)
            // Insert the number of results in the variadic result as a
            // constant.
            kernelOperands.push_back(rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(numRes)));

        for (auto op : operands)
            kernelOperands.push_back(op);

        return kernelOperands;
    }
};

/**
 * @brief Rewrites `daphne::CreateVariadicPackOp` to `LLVM::AllocaOp` to create
 * an array for the required number of occurrences of a variadic operand.
 */
class CreateVariadicPackOpLowering : public OpConversionPattern<daphne::CreateVariadicPackOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::CreateVariadicPackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Set the insertion point to the beginning of the function surrounding
        // this CreateVariadicPackOp (see comment on AllocaOp above).
        Block &fb = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>().getBody().front();
        rewriter.setInsertionPointToStart(&fb);

        Type contType = op.getRes().getType().dyn_cast<daphne::VariadicPackType>().getContainedType();
        Type convType = typeConverter->convertType(contType);
        rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
            op.getOperation(), LLVM::LLVMPointerType::get(convType),
            rewriter.create<arith::ConstantOp>(op->getLoc(), op.getNumElementsAttr()), 1);
        return success();
    }
};

/**
 * @brief Rewrites `daphne::StoreVariadicPackOp` to `LLVM::StoreOp` to store
 * an occurrence of a variadic operand to the respective position in an array
 * created by lowering `daphne::CreateVariadicPackOp`.
 */
class StoreVariadicPackOpLowering : public OpConversionPattern<daphne::StoreVariadicPackOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::StoreVariadicPackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        mlir::Location loc = op->getLoc();
        mlir::Value pack = adaptor.getOperands()[0];
        mlir::Value item = adaptor.getOperands()[1];
        auto elementType = pack.getType().cast<LLVM::LLVMPointerType>().getElementType();
        std::vector<Value> indices = {rewriter.create<arith::ConstantOp>(loc, op.getPosAttr())};
        auto addr = rewriter.create<LLVM::GEPOp>(loc, pack.getType(), pack, indices);
        Type itemType = item.getType();
        if (itemType != elementType) {
            if (llvm::isa<LLVM::LLVMPointerType>(elementType)) {
                if (itemType.isSignedInteger())
                    item = rewriter.create<LLVM::SExtOp>(loc, rewriter.getI64Type(), item);
                else if (itemType.isUnsignedInteger() || itemType.isSignlessInteger())
                    item = rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI64Type(), item);
                else if (llvm::isa<FloatType>(itemType)) {
                    item = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), item);
                    item = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI64Type(), item);
                } else {
                    throw ErrorHandler::compilerError(loc, "LowerToLLVMPass", "itemType is an unsupported type");
                }
                item = rewriter.create<LLVM::IntToPtrOp>(loc, elementType, item);
            } else {
                throw ErrorHandler::compilerError(loc, "LowerToLLVMPass",
                                                  "casting to a non-pointer type in "
                                                  "StoreVariadicPackOpLowering is not implemented yet");
            }
        }
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op.getOperation(), item, addr);
        return success();
    }
};

class ParForOpLowering : public OpConversionPattern<daphne::ParForOp> {
  public:
    explicit ParForOpLowering(TypeConverter &typeConverter, MLIRContext *context)
        : OpConversionPattern(typeConverter, context) {}

    using OpConversionPattern::OpConversionPattern;

    // TODO: REFACTORE ME
    LogicalResult matchAndRewrite(daphne::ParForOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        auto module = op->getParentOfType<ModuleOp>();
        auto ip = rewriter.saveInsertionPoint();

        // *****************************************************************************
        // Extraction of parfor op body region to an function.
        // Outer scope SSA values and induction variable are passed as function arguments
        // and replaced respectively in the body of created function.
        // *****************************************************************************
        rewriter.setInsertionPointToStart(module.getBody());
        static int idx = 0;
        std::string funcName = "parfor_body_" + std::to_string(idx++);
        auto opBodyArgs = op.getBodyStmt().front().getArguments();

        auto i64Type = typeConverter->convertType(rewriter.getIntegerType(64, true));
        auto i1Ty = IntegerType::get(getContext(), 1);
        auto ptrI1Ty = LLVM::LLVMPointerType::get(i1Ty);
        auto ptrPtrI1Ty = LLVM::LLVMPointerType::get(ptrI1Ty);

        std::vector<int64_t> indexLoopCarried = {};
        determinateIndexesOfLoopCarriedVariables(op, opBodyArgs.drop_front(1).drop_back(1), indexLoopCarried);

        // (ptr<ptr<i8>>, ptr<ptr<i8>>, i64, ptr<i1>) -> (void)
        auto funcType =
            LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(rewriter.getContext()), {/*output=*/ptrPtrI1Ty,
                                                                                         /*input=*/ptrPtrI1Ty,
                                                                                         /*induction variable=*/i64Type,
                                                                                         /*daphne context=*/ptrI1Ty});
        auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), funcName, funcType);
        // move first block
        mlir::Block &funcBlock = *llvmFuncOp.addEntryBlock();
        funcBlock.getOperations().splice(funcBlock.end(), op.getBodyStmt().front().getOperations());

        // Detach from old region and inline conditional blocks if present
        op.getBodyStmt().getBlocks().remove(&op.getBodyStmt().front());
        auto &blocks = op.getBodyStmt().getBlocks();
        llvmFuncOp.getBody().getBlocks().splice(llvmFuncOp.getBody().end(), blocks, blocks.begin(), blocks.end());
        llvmFuncOp->setAttr("parfor_inplace_rewrite_needed", rewriter.getUnitAttr());

        // Manual conversion of block arguments
        // cf dialect does not convert properly inside of the parfor body loop function
        if(!convertBlockArguments(llvmFuncOp)) 
            return failure();

        rewriter.setInsertionPointToStart(&funcBlock);
        auto locFunc = llvmFuncOp.getLoc();

        auto funcOutArg = llvmFuncOp.getArgument(0);
        auto funcInArg = llvmFuncOp.getArgument(1);
        auto ivArg = llvmFuncOp.getArgument(2);
        auto dctxArg = llvmFuncOp.getArgument(3);

        // handle loop induction variable
        opBodyArgs[0].replaceAllUsesWith(ivArg);
        // handle daphne context
        opBodyArgs[opBodyArgs.size() - 1].replaceAllUsesWith(dctxArg);
        // remove induction variable and daphne context
        opBodyArgs = opBodyArgs.drop_front(1).drop_back(1);
        // handle other arguments
        auto args = op.getArgs();
        unsigned index = 0;

        for (auto [i, blockArg] : llvm::enumerate(opBodyArgs)) {
            // number of arguments must be exactly the same as number of operands stored in args
            if (index >= args.size())
                llvm::report_fatal_error(
                    "ParForOp has more block arguments than operands, which is not defined behavior");

            // convert type to llvm type
            auto targetType = typeConverter->convertType(args[index].getType());
            if (!targetType)
                llvm::report_fatal_error("Cannot convert operand to LLVM type");

            // Get T* pointer from funcInArg[index]
            auto indexVal =
                rewriter.create<LLVM::ConstantOp>(locFunc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(index));
            auto gep = rewriter.create<LLVM::GEPOp>(locFunc, ptrPtrI1Ty, ptrPtrI1Ty, funcInArg, ValueRange{indexVal});
            auto rawPtr = rewriter.create<LLVM::LoadOp>(locFunc, ptrI1Ty, gep);

            // if the target type is already a pointer, we can directly load it
            // otherwise we need first to load the pointer of value and then load the value from that pointer
            auto convType = targetType;
            if (!convType.isa<LLVM::LLVMPointerType>())
                convType = LLVM::LLVMPointerType::get(convType);

            auto bitcasted = rewriter.create<LLVM::BitcastOp>(locFunc, convType, rawPtr);

            mlir::Value loaded = bitcasted;
            if (!targetType.isa<LLVM::LLVMPointerType>()) {
                loaded = rewriter.create<LLVM::LoadOp>(locFunc, targetType, loaded);
            }
            blockArg.replaceAllUsesWith(loaded);

            ++index;
        }
        // ********************************************************************
        // Write results in output array as CallKernelOp prescribes.          *
        // ********************************************************************
        auto returnOp = llvm::dyn_cast<daphne::ReturnOp>(llvmFuncOp.getBody().back().getTerminator());

        if (!returnOp || !llvm::isa<daphne::ReturnOp>(returnOp))
            llvm::report_fatal_error("Cannot determinate terminator of parfor body region, expected daphne::ReturnOp");

        rewriter.setInsertionPoint(returnOp);
        auto numRes = returnOp->getNumOperands();
        for (auto i = 0u; i < numRes; ++i) {
            auto retVal = returnOp->getOperand(i);
            auto loc = returnOp->getLoc();

            auto addrIdx = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i));
            auto gep = rewriter.create<LLVM::GEPOp>(loc, ptrPtrI1Ty, funcOutArg, ArrayRef<Value>({addrIdx}));
            auto llvmTy = typeConverter->convertType(retVal.getType());
            Value retValConverted = typeConverter->materializeTargetConversion(rewriter, loc, llvmTy, retVal);

            rewriter.create<LLVM::StoreOp>(loc, retValConverted, gep);
        }
        // Replace the old ReturnOp with operands by a new ReturnOp without
        // operands.
        rewriter.replaceOpWithNewOp<daphne::ReturnOp>(returnOp);

        // *******************************************************************************
        // * Create a ptr<ptr<i1>> array containg all outer scope operands of parfor op. *
        // * which is passed to the function created above as funcInArg and funcOutArg.  *
        // *******************************************************************************

        // TODO: alloca must probably be added on top of function (see comment about Alloca on top of the current file )
        // TODO: alternative - the same logic can be probably achieved by store variadic pack,
        // but this will also affect the dereference logic above.
        rewriter.restoreInsertionPoint(ip);
        unsigned numArgs = args.size();
        auto arraySizeConst = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(numArgs));
        auto arrayBaseVoidPtr = rewriter.create<LLVM::AllocaOp>(loc, ptrPtrI1Ty, arraySizeConst, 0);
        index = 0;
        auto one = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(1));

        for (Value operand : args) {
            auto operandType = operand.getType();
            auto llvmOperandType = typeConverter->convertType(operandType);

            // Ensure operand is converted to an LLVM-compatible type
            if (!LLVM::isCompatibleType(operandType)) {
                auto converted = typeConverter->materializeTargetConversion(rewriter, loc, llvmOperandType, operand);
                if (!converted)
                    llvm::report_fatal_error("Cannot convert operand to LLVM type");
                operand = converted;
            }

            Value operandAsPtr;

            if (llvmOperandType.isa<LLVM::LLVMPointerType>()) {
                // Already a pointer, cast to ptr<i1>
                operandAsPtr = rewriter.create<LLVM::BitcastOp>(loc, ptrI1Ty, operand);
            } else {
                // Allocate space for the value pointer and store it array
                auto alloca = rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(llvmOperandType), one, 0);
                rewriter.create<LLVM::StoreOp>(loc, operand, alloca);
                operandAsPtr = rewriter.create<LLVM::BitcastOp>(loc, ptrI1Ty, alloca);
            }
            auto gepIndex = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(index));
            auto gepPtr = rewriter.create<LLVM::GEPOp>(loc, ptrPtrI1Ty, arrayBaseVoidPtr, ValueRange{gepIndex});
            rewriter.create<LLVM::StoreOp>(loc, operandAsPtr, gepPtr);

            ++index;
        }

        // *****************************************************************************
        // Create a kernel call to the parfor loop body function
        // *****************************************************************************
        // TODO: we need probably to check if all individuall values of result array have the same type
        // but this is also done by CallKernelOp.
        auto fnPtr = rewriter.create<LLVM::AddressOfOp>(loc, llvmFuncOp);
        std::stringstream callee;
        auto kIdVal = rewriter.getI32IntegerAttr(KernelDispatchMapping::instance().registerKernel(funcName, op));
        auto kId = rewriter.create<mlir::arith::ConstantOp>(loc, kIdVal);

        auto loopCarriedTy = daphne::VariadicPackType::get(rewriter.getContext(), rewriter.getI64Type());
        auto loopCarriedArr =
            rewriter.create<daphne::CreateVariadicPackOp>(loc, loopCarriedTy, indexLoopCarried.size());
        for (auto [i, loopCarrIdx] : llvm::enumerate(indexLoopCarried)) {
            auto attrK = rewriter.getI64IntegerAttr(i);
            auto idxConst =
                rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(loopCarrIdx));
            rewriter.create<daphne::StoreVariadicPackOp>(loc, loopCarriedArr, idxConst, attrK);
        }

        llvm::errs() << "err \n";
        std::vector<Value> kernelOperands{
            adaptor.getFrom(), adaptor.getTo(), adaptor.getStep(), arrayBaseVoidPtr, fnPtr, loopCarriedArr, kId,
            adaptor.getCtx()};

        auto resultTypes = op->getResultTypes();
        if (numRes > 0) {
            auto firstResultTy = resultTypes[0];
            for (auto resultTy : resultTypes) {
                if (resultTy != firstResultTy) {
                    throw ErrorHandler::compilerError(loc, "LowerToLLVMPass",
                                                      "Different result types are not supported yet!");
                }
            }
            callee << "_parfor__" << CompilerUtils::mlirTypeToCppTypeName(firstResultTy, false) << "_variadic__size_t";
            callee << "__int64_t__int64_t__int64_t__void__void__int64_t";
        } else { // no results, so we need to add null ptr as output and 0 as number of results values as kernel
                 // operands
            callee << "_parfor__void_variadic__size_t__int64_t__int64_t__int64_t__void__void__int64_t";
            kernelOperands.insert(kernelOperands.begin(), rewriter.create<LLVM::NullOp>(loc, ptrPtrI1Ty));
            kernelOperands.insert(kernelOperands.begin() + 1,
                                  rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0)));
        }
        auto kernel = rewriter.create<daphne::CallKernelOp>(loc, callee.str(), kernelOperands, resultTypes);
        if (numRes > 0)
            kernel->setAttr(ATTR_HASVARIADICRESULTS, rewriter.getBoolAttr(true));
        rewriter.replaceOp(op, kernel.getResults());

        // module.dump();
        // exit(-1);
        return success();
    }

    bool convertBlockArguments(LLVM::LLVMFuncOp func) const {
        for (Block &block : func.getBody().getBlocks()) {
            for (auto &arg : block.getArguments()) {
                auto newType = typeConverter->convertType(arg.getType());
                if (!newType)
                    return false;
                arg.setType(newType);
            }
        }
        return true;
    }

    void determinateIndexesOfLoopCarriedVariables(daphne::ParForOp op, mlir::Block::BlockArgListType args,
                                                  std::vector<int64_t> &indexLoopCarried) const {
        auto returnOp = llvm::dyn_cast<daphne::ReturnOp>(op.getBodyStmt().back().getTerminator());
        if (!returnOp || returnOp.getNumOperands() == 0) {
            return; // No operands to analyze, i.e. there is no reduction to conduct after parellel computation
        }
        for (auto [idx, arg] : llvm::enumerate(args)) {
            Value *returnVal = nullptr;
            if (isLoopCarried(arg, returnOp, returnVal)) {
                if (returnVal != nullptr) {
                    indexLoopCarried.push_back(idx);
                }
            }
        }
    }

    /**
     * @brief Checks if the given block argument is loop-carried by checking if
     * it is used in the return operation of the parfor body region.
     *
     * @param arg The block argument to check.
     * @param returnOp The return operation of the parfor body region.
     * @param returnVal A reference to store the value if it is loop-carried.
     * @return true if the argument is loop-carried, false otherwise.
     */
    bool isLoopCarried(BlockArgument arg, daphne::ReturnOp returnOp, Value *&returnVal) const {
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

class MapOpLowering : public OpConversionPattern<daphne::MapOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::MapOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        auto module = op->getParentOfType<ModuleOp>();

        std::stringstream callee;
        callee << '_' << op->getName().stripDialect().str();

        // Result Matrix
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(op.getType(), false);

        // Input Matrix
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(op.getArg().getType(), false);

        // Pointer to UDF
        callee << "__void";

        // get pointer to UDF
        LLVM::LLVMFuncOp udfFuncOp = module.lookupSymbol<LLVM::LLVMFuncOp>(op.getFunc());
        auto udfFnPtr = rewriter.create<LLVM::AddressOfOp>(loc, udfFuncOp);

        std::vector<Value> kernelOperands{op.getArg(), udfFnPtr};

        auto kernel = rewriter.create<daphne::CallKernelOp>(loc, callee.str(), kernelOperands, op->getResultTypes());
        rewriter.replaceOp(op, kernel.getResults());

        return success();
    }
};

class VectorizedPipelineOpLowering : public OpConversionPattern<daphne::VectorizedPipelineOp> {
    const DaphneUserConfig &cfg;

  public:
    explicit VectorizedPipelineOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                                          const DaphneUserConfig &cfg)
        : OpConversionPattern(typeConverter, context), cfg(cfg) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::VectorizedPipelineOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (op.getCtx() == nullptr) {
            op->emitOpError() << "`DaphneContext` not known";
            return failure();
        }
        auto loc = op->getLoc();
        auto numDataOperands = op.getInputs().size();
        std::vector<mlir::Value> func_ptrs;

        auto i1Ty = IntegerType::get(getContext(), 1);
        auto ptrI1Ty = LLVM::LLVMPointerType::get(i1Ty);
        auto ptrPtrI1Ty = LLVM::LLVMPointerType::get(ptrI1Ty);
        auto pppI1Ty = LLVM::LLVMPointerType::get(ptrPtrI1Ty);

        std::string vecFuncName = "_vect";
        LLVM::LLVMFuncOp fOp;
        {
            OpBuilder::InsertionGuard ig(rewriter);
            auto moduleOp = op->getParentOfType<ModuleOp>();
            Block *moduleBody = moduleOp.getBody();
            rewriter.setInsertionPointToStart(moduleBody);

            static auto ix = 0;
            std::string funcName = "_vect" + std::to_string(++ix);
            vecFuncName = funcName;

            // TODO: pass daphne context to function
            auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(rewriter.getContext()),
                                                        {/*outputs...*/ pppI1Ty, /*inputs...*/ ptrPtrI1Ty,
                                                         /*daphneContext...*/ ptrI1Ty});

            fOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
            fOp.getBody().takeBody(op.getBody());
            auto &funcBlock = fOp.getBody().front();

            auto returnRef = funcBlock.addArgument(pppI1Ty, rewriter.getUnknownLoc());
            auto inputsArg = funcBlock.addArgument(ptrPtrI1Ty, rewriter.getUnknownLoc());
            auto daphneContext = funcBlock.addArgument(ptrI1Ty, rewriter.getUnknownLoc());

            for (auto callKernelOp : funcBlock.getOps<daphne::CallKernelOp>()) {
                callKernelOp.setOperand(callKernelOp.getNumOperands() - 1, daphneContext);
            }

            // Extract inputs from array containing them and remove the block
            // arguments matching the old inputs of the `VectorizedPipelineOp`
            rewriter.setInsertionPointToStart(&funcBlock);

            for (auto i = 0u; i < numDataOperands; ++i) {
                auto addr = rewriter.create<LLVM::GEPOp>(
                    loc, ptrPtrI1Ty, inputsArg,
                    ArrayRef<Value>({rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i))}));
                Value val = rewriter.create<LLVM::LoadOp>(loc, addr);
                auto expTy = typeConverter->convertType(op.getInputs().getType()[i]);
                if (expTy != val.getType()) {
                    // casting for scalars
                    val = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), val);
                    if (llvm::isa<IntegerType>(expTy))
                        val = rewriter.create<LLVM::TruncOp>(loc, expTy, val);
                    else if (llvm::isa<FloatType>(expTy)) {
                        val = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF64Type(), val);
                        val = rewriter.create<LLVM::FPTruncOp>(loc, expTy, val);
                    } else {
                        throw ErrorHandler::compilerError(loc, "LowerToLLVMPass", "expTy is an unsupported type");
                    }
                }
                funcBlock.getArgument(0).replaceAllUsesWith(val);
                funcBlock.eraseArgument(0);
            }

            // Update function block to write return value by reference instead
            auto *oldReturn = funcBlock.getTerminator();
            rewriter.setInsertionPoint(oldReturn);
            for (auto i = 0u; i < oldReturn->getNumOperands(); ++i) {
                auto retVal = oldReturn->getOperand(i);
                // TODO: check how the GEPOp works exactly, and if this can be
                // written better
                auto addr1 = rewriter.create<LLVM::GEPOp>(
                    op->getLoc(), pppI1Ty, returnRef,
                    ArrayRef<Value>({rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i))}));
                auto addr2 = rewriter.create<LLVM::LoadOp>(op->getLoc(), addr1);
                Value retValConverted = typeConverter->materializeTargetConversion(
                    rewriter, oldReturn->getLoc(), typeConverter->convertType(retVal.getType()), {retVal});
                rewriter.create<LLVM::StoreOp>(loc, retValConverted, addr2);
            }
            // Replace the old ReturnOp with operands by a new ReturnOp without
            // operands.
            rewriter.replaceOpWithNewOp<func::ReturnOp>(oldReturn);
        }

        auto fnPtr = rewriter.create<LLVM::AddressOfOp>(loc, fOp);

        func_ptrs.push_back(fnPtr);

        if (cfg.use_cuda && !op.getCuda().getBlocks().empty()) {
            LLVM::LLVMFuncOp fOp2;
            {
                OpBuilder::InsertionGuard ig(rewriter);
                auto moduleOp = op->getParentOfType<ModuleOp>();
                Block *moduleBody = moduleOp.getBody();
                rewriter.setInsertionPointToStart(moduleBody);

                static auto ix = 0;
                std::string funcName = "_vect_cuda" + std::to_string(++ix);

                auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(rewriter.getContext()),
                                                            {/*outputs...*/ pppI1Ty, /*inputs...*/ ptrPtrI1Ty,
                                                             /*daphneContext...*/ ptrI1Ty});

                fOp2 = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
                fOp2.getBody().takeBody(op.getCuda());
                auto &funcBlock = fOp2.getBody().front();

                auto returnRef = funcBlock.addArgument(pppI1Ty, rewriter.getUnknownLoc());
                auto inputsArg = funcBlock.addArgument(ptrPtrI1Ty, rewriter.getUnknownLoc());
                auto daphneContext = funcBlock.addArgument(ptrI1Ty, rewriter.getUnknownLoc());

                // TODO: we should not create a new daphneContext, instead pass
                // the one created in the main function
                for (auto callKernelOp : funcBlock.getOps<daphne::CallKernelOp>()) {
                    callKernelOp.setOperand(callKernelOp.getNumOperands() - 1, daphneContext);
                }

                // Extract inputs from array containing them and remove the
                // block arguments matching the old inputs of the
                // `VectorizedPipelineOp`
                rewriter.setInsertionPointToStart(&funcBlock);

                for (auto i = 0u; i < numDataOperands; ++i) {
                    auto addr = rewriter.create<LLVM::GEPOp>(
                        loc, ptrPtrI1Ty, inputsArg,
                        ArrayRef<Value>({rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i))}));
                    Value val = rewriter.create<LLVM::LoadOp>(loc, addr);
                    auto expTy = typeConverter->convertType(op.getInputs().getType()[i]);
                    if (expTy != val.getType()) {
                        val = rewriter.create<LLVM::PtrToIntOp>(
                            loc, rewriter.getIntegerType(expTy.getIntOrFloatBitWidth(), false), val);
                        val = rewriter.create<LLVM::BitcastOp>(loc, expTy, val);
                    }
                    funcBlock.getArgument(0).replaceAllUsesWith(val);
                    funcBlock.eraseArgument(0);
                }

                // Update function block to write return value by reference
                // instead
                auto *oldReturn = funcBlock.getTerminator();
                rewriter.setInsertionPoint(oldReturn);
                for (auto i = 0u; i < oldReturn->getNumOperands(); ++i) {
                    auto retVal = oldReturn->getOperand(i);
                    // TODO: check how the GEPOp works exactly, and if this can
                    // be written better
                    auto addr1 = rewriter.create<LLVM::GEPOp>(
                        op->getLoc(), pppI1Ty, returnRef,
                        ArrayRef<Value>({rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i))}));
                    auto addr2 = rewriter.create<LLVM::LoadOp>(op->getLoc(), addr1);
                    Value retValConverted = typeConverter->materializeTargetConversion(
                        rewriter, oldReturn->getLoc(), typeConverter->convertType(retVal.getType()), {retVal});
                    rewriter.create<LLVM::StoreOp>(loc, retValConverted, addr2);
                }
                // Replace the old ReturnOp with operands by a new ReturnOp
                // without operands.
                rewriter.replaceOpWithNewOp<func::ReturnOp>(oldReturn);
            }

            auto fnPtr2 = rewriter.create<LLVM::AddressOfOp>(loc, fOp2);

            func_ptrs.push_back(fnPtr2);
        }
        std::stringstream callee;
        callee << '_' << op->getName().stripDialect().str();

        // Get some information on the results.
        Operation::result_type_range resultTypes = op->getResultTypes();
        const size_t numRes = op->getNumResults();

        if (numRes > 0) {
            // TODO Support individual types for all outputs (see #397).
            // Check if all results have the same type.
            Type mt0 = resultTypes[0].dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr();
            for (size_t i = 1; i < numRes; i++) {
                if (mt0 != resultTypes[i].dyn_cast<daphne::MatrixType>().withSameElementTypeAndRepr()) {
                    throw ErrorHandler::compilerError(op, "LowerToLLVMPass",
                                                      "encountered a vectorized pipelines with different "
                                                      "result types, but at the moment we require all "
                                                      "results to have the same type");
                }
            }
            // Append the name of the common type of all results to the kernel
            // name.
            callee << "__" << CompilerUtils::mlirTypeToCppTypeName(resultTypes[0], false) << "_variadic__size_t";
        }

        mlir::Type operandType;
        std::vector<Value> newOperands;
        if (numRes > 0) {
            auto m32type = rewriter.getF32Type();
            auto m64type = rewriter.getF64Type();
            auto msi64type = rewriter.getIntegerType(64, true);

            auto res_elem_type = op->getResult(0).getType().dyn_cast<mlir::daphne::MatrixType>().getElementType();
            if (res_elem_type == m64type)
                operandType = daphne::MatrixType::get(getContext(), m64type);
            else if (res_elem_type == m32type)
                operandType = daphne::MatrixType::get(getContext(), m32type);
            else if (res_elem_type == msi64type)
                operandType = daphne::MatrixType::get(getContext(), msi64type);
            else {
                std::string str;
                llvm::raw_string_ostream output(str);
                op->getResult(0).getType().print(output);
                throw ErrorHandler::compilerError(op, "LowerToLLVMPass",
                                                  "Unsupported result type for vectorizedPipeline op: " + str);
            }
        } else {
            throw ErrorHandler::compilerError(op, "LowerToLLVMPass",
                                              "vectorizedPipelineOp without outputs not supported at the "
                                              "moment!");
        }

        // Handle variadic operands isScalar and inputs (both share numInputs).
        auto attrNumInputs = rewriter.getI64IntegerAttr(numDataOperands);
        // For isScalar.
        callee << "__bool";
        auto vpScalar = rewriter.create<daphne::CreateVariadicPackOp>(
            loc, daphne::VariadicPackType::get(rewriter.getContext(), rewriter.getI1Type()), attrNumInputs);
        // For inputs and numInputs.
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(operandType, false, true);
        callee << "_variadic__size_t";
        auto vpInputs = rewriter.create<daphne::CreateVariadicPackOp>(
            loc, daphne::VariadicPackType::get(rewriter.getContext(), operandType), attrNumInputs);
        // Populate the variadic packs for isScalar and inputs.
        for (size_t k = 0; k < numDataOperands; k++) {
            auto attrK = rewriter.getI64IntegerAttr(k);
            rewriter.create<daphne::StoreVariadicPackOp>(
                loc, vpScalar,
                rewriter.create<daphne::ConstantOp>(
                    loc,
                    // We assume this input to be a scalar if its type
                    // has not been converted to a pointer type.
                    !llvm::isa<LLVM::LLVMPointerType>(adaptor.getOperands()[k].getType())),
                attrK);
            rewriter.create<daphne::StoreVariadicPackOp>(loc, vpInputs, adaptor.getOperands()[k], attrK);
        }
        newOperands.push_back(vpScalar);
        newOperands.push_back(vpInputs);
        newOperands.push_back(
            rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(numDataOperands)));

        // Obtain an insertion point at the beginning of the function
        // surrounding this VectorizedPipelineOp (see comment on AllocaOp
        // above).
        OpBuilder::InsertPoint ipHere = rewriter.saveInsertionPoint();
        Block &fb = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>().getBody().front();
        rewriter.setInsertionPointToStart(&fb);
        OpBuilder::InsertPoint ipFuncStart = rewriter.saveInsertionPoint();
        rewriter.restoreInsertionPoint(ipHere);

        auto numOutputs = op.getNumResults();
        // Variadic num rows operands.
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(rewriter.getIntegerType(64, true), false);
        auto rowsOperands = adaptor.getOperands().drop_front(numDataOperands);
        newOperands.push_back(
            convertToArray(loc, rewriter, rewriter.getI64Type(), rowsOperands.take_front(numOutputs), ipFuncStart));
        callee << "__" << CompilerUtils::mlirTypeToCppTypeName(rewriter.getIntegerType(64, true), false);
        auto colsOperands = rowsOperands.drop_front(numOutputs);
        newOperands.push_back(
            convertToArray(loc, rewriter, rewriter.getI64Type(), colsOperands.take_front(numOutputs), ipFuncStart));

        // Add array of split enums
        callee << "__int64_t";
        std::vector<Value> splitConsts;
        for (auto split : op.getSplits()) {
            splitConsts.push_back(rewriter.create<arith::ConstantOp>(loc, split));
        }
        newOperands.push_back(convertToArray(loc, rewriter, rewriter.getI64Type(), splitConsts, ipFuncStart));

        // Add array of combine enums
        callee << "__int64_t";
        std::vector<Value> combineConsts;
        for (auto combine : op.getCombines()) {
            combineConsts.push_back(rewriter.create<arith::ConstantOp>(loc, combine));
        }
        newOperands.push_back(convertToArray(loc, rewriter, rewriter.getI64Type(), combineConsts, ipFuncStart));

        // TODO: pass function pointer with special placeholder instead of
        // `void`

        callee << "__size_t";
        newOperands.push_back(
            rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(func_ptrs.size())));
        callee << "__void_variadic";
        newOperands.push_back(convertToArray(loc, rewriter, ptrPtrI1Ty, func_ptrs, ipFuncStart));
        //        newOperands.push_back(fnPtr);

        // Add ctx
        //        newOperands.push_back(operands.back());
        if (op.getCtx() == nullptr) {
            // TODO(phil): make ErrorHandler call
            op->emitOpError() << "`DaphneContext` not known";
            return failure();
        }

        auto kId = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(KernelDispatchMapping::instance().registerKernel(vecFuncName, op)));

        newOperands.push_back(kId);
        newOperands.push_back(op.getCtx());

        // Create a CallKernelOp for the kernel function to call and return
        // success().
        auto kernel = rewriter.create<daphne::CallKernelOp>(loc, callee.str(), newOperands, resultTypes);
        kernel->setAttr(ATTR_HASVARIADICRESULTS, rewriter.getBoolAttr(true));
        rewriter.replaceOp(op, kernel.getResults());
        return success();
    }

  private:
    static Value convertToArray(Location loc, ConversionPatternRewriter &rewriter, Type valueTy, ValueRange values,
                                OpBuilder::InsertPoint &ipFuncStart) {
        // Set the insertion point to the beginning of the function surrounding
        // this VectorizedPipelineOp (see comment on AllocaOp above).
        OpBuilder::InsertPoint ipHere = rewriter.saveInsertionPoint();
        rewriter.restoreInsertionPoint(ipFuncStart);

        auto valuePtrTy = LLVM::LLVMPointerType::get(valueTy);
        auto array = rewriter.create<LLVM::AllocaOp>(
            loc, valuePtrTy, Value(rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(values.size()))));
        ipFuncStart = rewriter.saveInsertionPoint();

        // Go back to the original insertion point.
        rewriter.restoreInsertionPoint(ipHere);

        for (auto i = 0u; i < values.size(); ++i) {
            Value cstI = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i));
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

class GenericCallOpLowering : public OpConversionPattern<daphne::GenericCallOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(daphne::GenericCallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), op->getResultTypes(), adaptor.getOperands());
        return success();
    }
};

namespace {
struct DaphneLowerToLLVMPass : public PassWrapper<DaphneLowerToLLVMPass, OperationPass<ModuleOp>> {
    explicit DaphneLowerToLLVMPass(const DaphneUserConfig &cfg) : cfg(cfg) {}
    const DaphneUserConfig &cfg;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect /*, scf::SCFDialect*/>();
    }
    void runOnOperation() final;
};
} // end anonymous namespace

void DaphneLowerToLLVMPass::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    RewritePatternSet postConversionPatterns(&getContext());

    LowerToLLVMOptions llvmOptions(&getContext());
    // llvmOptions.useBarePtrCallConv = true;
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);
    typeConverter.addConversion(
        [&](daphne::MatrixType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::FrameType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::ListType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::ColumnType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::StringType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 8)); });
    typeConverter.addConversion([&](daphne::VariadicPackType t) {
        return LLVM::LLVMPointerType::get(typeConverter.convertType(t.getContainedType()));
    });
    typeConverter.addConversion(
        [&](daphne::DaphneContextType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::HandleType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::FileType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::DescriptorType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });
    typeConverter.addConversion(
        [&](daphne::TargetType t) { return LLVM::LLVMPointerType::get(IntegerType::get(t.getContext(), 1)); });

    LLVMConversionTarget target(getContext());

    // populate dialect conversions
    mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    target.addLegalOp<ModuleOp>();
    // for trivial casts no lowering to kernels -> higher benefit
    patterns.insert<CastOpLowering>(&getContext(), 2);
    patterns.insert<CallKernelOpLowering, CreateVariadicPackOpLowering>(typeConverter, &getContext());
    patterns.insert<VectorizedPipelineOpLowering>(typeConverter, &getContext(), cfg);
    patterns.insert<ConstantOpLowering, ReturnOpLowering, StoreVariadicPackOpLowering, GenericCallOpLowering,
                    MapOpLowering>(&getContext());

    patterns.insert<ParForOpLowering>(typeConverter, &getContext());

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createLowerToLLVMPass(const DaphneUserConfig &cfg) {
    return std::make_unique<DaphneLowerToLLVMPass>(cfg);
}
