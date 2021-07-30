# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_mlir_dialect(DaphneOps daphne) # creates the target MLIRDaphneOpsIncGen
add_mlir_doc(DaphneOps -gen-dialect-doc DaphneDialect Dialects/)

add_mlir_interface(DaphneDistributableOpInterface)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Daphne)
add_public_tablegen_target(MLIRDaphneTransformsIncGen)

add_mlir_doc(Passes -gen-pass-doc DaphnePasses Dialects/)

add_mlir_dialect_library(MLIRDaphne
    DaphneDialect.cpp

    DEPENDS
    MLIRDaphneOpsIncGen
    MLIRDaphneTransformsIncGen

    LINK_COMPONENTS
    Core
)