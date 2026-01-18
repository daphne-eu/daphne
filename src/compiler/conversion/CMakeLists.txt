add_mlir_conversion_library(MLIRDaphneToLinalg
    DaphneToLinalg.cpp
    DaphneToLinalgPass.cpp

    DEPENDS
    MLIRDaphneOpsIncGen
    MLIRDaphneTransformsIncGen

    LINK_LIBS PUBLIC
    CompilerUtils

    LINK_COMPONENTS
    Core

    )
