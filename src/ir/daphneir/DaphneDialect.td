/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD
#define SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD

include "mlir/IR/OpBase.td"

// ****************************************************************************
// Daphne dialect definition.
// ****************************************************************************

def Daphne_Dialect : Dialect {
    let name = "daphne";
    let summary = "An MLIR dialect for DAPHNE.";
    let description = [{
      This is an MLIR dialect for DAPHNE. Details coming soon.
    }];
    let cppNamespace = "::mlir::daphne";

    let hasConstantMaterializer = 1;

    // TODO It should not be necessary to add the declarations of parseType()
    // and printType() manually. However, I cannot find a way to have only the
    // declarations (but not the definitions) generated with this version of
    // MLIR. Using useDefaultTypePrinterParser messes things up.
    let extraClassDeclaration = [{
        /// Parse a type registered to this dialect.
        ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

        /// Print a type registered to this dialect.
        void printType(::mlir::Type type,
                      ::mlir::DialectAsmPrinter &os) const override;
    }];
}

#endif //SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD