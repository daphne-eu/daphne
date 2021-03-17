#ifndef INCLUDE_MLIR_DIALECT_DAPHNE_PASSES_H
#define INCLUDE_MLIR_DIALECT_DAPHNE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace daphne
    {
        std::unique_ptr<Pass> createLowerToLLVMPass();
        std::unique_ptr<Pass> createRewriteToCallKernelOpPass();
    } // namespace daphne
} // namespace mlir

#endif //INCLUDE_MLIR_DIALECT_DAPHNE_PASSES_H