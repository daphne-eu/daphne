#ifndef SRC_IR_DAPHNEIR_PASSES_H
#define SRC_IR_DAPHNEIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace daphne
    {
        std::unique_ptr<Pass> createLowerToLLVMPass();
        std::unique_ptr<Pass> createRewriteToCallKernelOpPass();
    } // namespace daphne
} // namespace mlir

#endif //SRC_IR_DAPHNEIR_PASSES_H