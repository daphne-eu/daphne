add_mlir_dialect_library(MLIRDaphneTransforms
    RewriteToCallKernelOpPass.cpp
    LowerToLLVMPass.cpp

    DEPENDS
    MLIRDaphneOpsIncGen

    LINK_COMPONENTS
    Core
)