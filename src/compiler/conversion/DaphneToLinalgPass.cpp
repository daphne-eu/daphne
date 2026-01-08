#include "compiler/conversion/DaphneToLinalg.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "compiler/utils/DaphneTypeConverter.h"
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

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
        target.addLegalDialect<tensor::TensorDialect, bufferization::BufferizationDialect, linalg::LinalgDialect,
                               arith::ArithDialect, memref::MemRefDialect, func::FuncDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();

        target.addIllegalOp<daphne::FillOp, daphne::AllAggSumOp, daphne::AllAggMinOp, daphne::AllAggMaxOp,
                            daphne::EwAddOp, daphne::EwSubOp, daphne::EwMulOp, daphne::EwDivOp, daphne::ReturnOp>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, memref::MemRefDialect, linalg::LinalgDialect, tensor::TensorDialect,
                        bufferization::BufferizationDialect, sparse_tensor::SparseTensorDialect>();
    }
};

std::unique_ptr<Pass> createConvertDaphneToLinalgPass() { return std::make_unique<ConvertDaphneToLinalgPass>(); }
} // namespace mlir::daphne
