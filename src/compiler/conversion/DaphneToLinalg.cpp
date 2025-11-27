#include "compiler/conversion/DaphneToLinalg.h"
#include "ir/daphneir/Daphne.h"
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

struct FillOpConverter : public OpConversionPattern<daphne::FillOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(daphne::FillOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        op.dump();
        op.getType().dump();
        adaptor.getArg().dump();
      auto mt = llvm::dyn_cast<daphne::MatrixType>(op.getType());
      if (!mt)
        return rewriter.notifyMatchFailure(op, "expected MatrixType result");

      auto memrefTy = llvm::dyn_cast<MemRefType>(typeConverter->convertType(mt));
      if (!memrefTy)
        return rewriter.notifyMatchFailure(op, "no memref type after conversion");

      // Collect dynamic dims for alloc
      SmallVector<Value, 2> dynSizes;
      if (memrefTy.isDynamicDim(0))
        dynSizes.push_back(adaptor.getNumRows());
      if (memrefTy.isDynamicDim(1))
        dynSizes.push_back(adaptor.getNumCols());

      Location loc = op.getLoc();
      Value alloc = rewriter.create<memref::AllocOp>(loc, memrefTy, dynSizes);

      // linalg.fill on buffers writes into the memref; use the output as replacement.
      rewriter.create<linalg::FillOp>(loc, adaptor.getArg(), alloc);
      rewriter.replaceOp(op, alloc);
      return success();
    }
};

void populateDaphneToLinalgPatterns(DaphneTypeConverter &converter, RewritePatternSet &patterns) {
    patterns.add<FillOpConverter>(converter, patterns.getContext());
}
