#include "compiler/conversion/DaphneToLinalg.h"
#include "ir/daphneir/Passes.h"

#include "compiler/utils/DaphneTypeConverter.h"
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace mlir::daphne {
#define GEN_PASS_DEF_CONVERTDAPHNETOLINALG
#include "ir/daphneir/Passes.h.inc"

using namespace mlir;

struct ConvertDaphneToLinalgPass : public impl::ConvertDaphneToLinalgBase<ConvertDaphneToLinalgPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        DaphneTypeConverter converter(&getContext());
        populateDaphneToLinalgPatterns(converter, patterns);

        ConversionTarget target(getContext());
        target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, memref::MemRefDialect>();
        target.addIllegalOp<daphne::FillOp>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> createConvertDaphneToLinalgPass() { return std::make_unique<ConvertDaphneToLinalgPass>(); }
} // namespace mlir::daphne
