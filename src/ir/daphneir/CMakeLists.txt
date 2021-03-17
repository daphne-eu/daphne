add_mlir_dialect(DaphneOps daphne) # creates the target MLIRDaphneOpsIncGen
add_mlir_doc(DaphneOps -gen-dialect-doc DaphneoDialect Dialects/)

add_mlir_dialect_library(MLIRDaphne
    DaphneDialect.cpp

    DEPENDS
    MLIRDaphneOpsIncGen

    LINK_COMPONENTS
    Core
)