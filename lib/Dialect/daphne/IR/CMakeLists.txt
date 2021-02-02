add_mlir_dialect_library(MLIRDaphne
    DaphneDialect.cpp

    DEPENDS
    MLIRDaphneOpsIncGen

    LINK_COMPONENTS
    Core
)