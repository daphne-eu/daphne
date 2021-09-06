#!/usr/bin/env python3

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

"""
Generates the C++ code for the pre-compiled kernels library.

This script generates C++ code instantiating the kernel templates that shall be
part of a pre-compiled kernel library. Each kernel instantiation is wrapped by
a shallow function that can be called from the JIT-compiled user program. An
input JSON-file specifies which kernel shall be instantiated with which
template arguments.
"""

# TODO Note that this script currently makes strong assumptions about the
# signatures of possible kernels. It might not yet generalize well enough.

# TODO Note that the way the information on the kernel template is specified in
# the input JSON-file will be simplified significantly later on.

import json
import sys

INDENT = 4 * " "
DEFAULT_NEWRESPARAM = "res"

def toCppType(t):
    return "{}<{}>".format(t[0], t[1]) if isinstance(t, list) else t

def generateKernelInstantiation(kernelTemplateInfo, templateValues, opCodes, outFile):
    # Extract some information.
    opName = kernelTemplateInfo["opName"]
    returnType = kernelTemplateInfo["returnType"]
    templateParams = kernelTemplateInfo["templateParams"]
    runtimeParams = kernelTemplateInfo["runtimeParams"]
    if opCodes is not None:# and opName != "PoolForward":
        # We assume that the op-code is the first run-time parameter.
        opCodeType = runtimeParams[0]["type"]
        runtimeParams = runtimeParams[1:]
    else:
        opCodeType = None

    # Create mapping from original template argument names to assigned C++
    # types.
    templateArgToCppType = {tp["name"]: toCppType(tv) for tp, tv in zip(templateParams, templateValues)}

    # Comments indicating values assigned to original template arguments.
    for tp in templateParams:
        outFile.write(INDENT + "// {} = {}\n".format(tp["name"], templateArgToCppType[tp["name"]]))
    
    # The function wrapping the generated kernel instantiation always has
    # the return type void. If the considered kernel returns a scalar value,
    # we prepend an additional run-time parameter. 
    extendedRuntimeParams = [
        {"name": DEFAULT_NEWRESPARAM, "type": "{} *".format(returnType), "isOutput": True}
    ] if (returnType != "void") else []
    # Add all run-time parameters of the kernel. We need to copy, because
    # we apply string replacements to the types.
    extendedRuntimeParams.extend([rp.copy() for rp in runtimeParams])
    # Replace occurences of original template arguments by their assigned
    # types.
    for rp in extendedRuntimeParams:
        for tpIdx, tp in enumerate(templateParams):
            if isinstance(templateValues[tpIdx], list):
                rp["type"] = rp["type"].replace("typename {}::VT".format(tp["name"]), templateValues[tpIdx][1])
            rp["type"] = rp["type"].replace(tp["name"], templateArgToCppType[tp["name"]])
        if rp["type"].endswith("*&"):
            rp["type"] = rp["type"][:-2] + "**"
            rp["isOutput"] = True
        elif "isOutput" not in rp:
            rp["isOutput"] = False

    isCreateDaphneContext = opName == "createDaphneContext"

    #typesForName = "__".join([("{}_{}".format(tv[0], tv[1]) if isinstance(tv, list) else tv) for tv in templateValues])
    typesForName = "__".join([
        rp["type"]
        .replace("const ", "")
        .replace(" **", "" if rp["isOutput"] else "_variadic")
        .replace(" *", "")
        .replace("& ", "")
        .replace("<", "_").replace(">", "")
        for rp in extendedRuntimeParams
    ])
    if typesForName != "":
        typesForName = "__" + typesForName
    params = ", ".join(
            ["{} {}".format(rtp["type"], rtp["name"]) for rtp in extendedRuntimeParams] +
            ([] if isCreateDaphneContext else ["DCTX(ctx)"])
    )

    def generateFunction(opCode):
        # Obtain the name of the function to be generated from the opName by
        # removing suffices "Sca"/"Mat"/"Obj" (they are not required here), and
        # potentially by inserting the opCode into the name.
        funcName = "_" + opName
        while funcName[-3:] in ["Sca", "Mat", "Obj"]:
            funcName = funcName[:-3]
        if opCode is not None:
            # We assume that the name of the op-code type ends with "OpCode".
            if funcName == "_PoolForward":
                funcName = "_" + opCode.lower() + funcName[1:]
            else:
                opCodeWord = opCodeType[:-len("OpCode")]
                funcName = funcName.replace(opCodeWord, opCode[0].upper() + opCode[1:].lower())
                funcName = funcName.replace(opCodeWord.lower(), opCode.lower())

        # Signature of the function wrapping the kernel instantiation.
        outFile.write(INDENT + "void {}{}({}) {{\n".format(
                funcName,
                typesForName,
                # Run-time parameters, possibly including DaphneContext:
                params
        ))
        
        # List of parameters for the call.
        if opCode is None:# and funcName != "PoolForward":
            callParams = []
        else:
            callParams = ["{}::{}".format(opCodeType, opCode)]
        callParams.extend([
            # Dereference double pointer for output parameters.
            "{}{}".format("*" if (rp["type"].endswith("**") and rp["isOutput"]) else "", rp["name"])
            for rp
            in extendedRuntimeParams[(0 if returnType == "void" else 1):]
        ])
        
        # List of template parameters for the call.
        callTemplateParams = [toCppType(tv) for tv in templateValues]
        
        # Body of that function: delegate to the kernel instantiation.
        outFile.write(2 * INDENT)
        if returnType != "void":
            outFile.write("*{} = ".format(DEFAULT_NEWRESPARAM))

        # to avoid compilation warnings
        if opName == "ewBinarySca" and opCode == "MUL":
            # handle bool return value with a cast and a uin32_t template parameter
            outFile.write(("{}<{}>::apply({});\n" if templateValues[0] != "bool" else "static_cast<bool>({}<{}>::apply({}));\n").format(
                "EwBinarySca",
                # Template parameters:
                ", ".join(["BinaryOpCode::MUL"] + ([toCppType(tv) for tv in templateValues] if templateValues[0] != "bool" else (["uint32_t"] + [toCppType(tv) for tv in templateValues[1:]]))),
                # Run-time parameters:
                ", ".join(callParams[1:] + ([] if isCreateDaphneContext else ["ctx"] )),
            ))
        elif opName == "PoolForward":
            outFile.write("{}<{}>::apply({});\n".format(
                "Pooling::Forward",
                # Template parameters:
                ", ".join(["Pooling::"+opCode]+[toCppType(tv) for tv in templateValues]),
                # Run-time parameters:
                # ", ".join(([] if isCreateDaphneContext else ["ctx"] ) + callParams[1:]),
                ", ".join(callParams[1:] + ([] if isCreateDaphneContext else ["ctx"] )),
            ))
        else:
            outFile.write("{}{}({});\n".format(
                    opName,
                    # Template parameters, if the kernel is a template:
                    "<{}>".format(", ".join(callTemplateParams)) if len(templateValues) else "",
                    # Run-time parameters, possibly including DaphneContext:
                    ", ".join(callParams + ([] if isCreateDaphneContext else ["ctx"] )),
            ))
        outFile.write(INDENT + "}\n")
    
    # Generate the function(s).
    if opCodes is None:
        generateFunction(None)
    else:
        for opCode in opCodes:
            generateFunction(opCode)
    outFile.write(INDENT + "\n")

def printHelp():
    print("Usage: python3 {} INPUT_SPEC_FILE OUTPUT_CPP_FILE".format(sys.argv[0]))
    print(__doc__)

if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        printHelp()
        sys.exit(0)
    elif len(sys.argv) != 3:
        print("Wrong number of arguments.")
        print()
        printHelp()
        sys.exit(1)

    # Parse arguments.
    inFilePath = sys.argv[1]
    outFilePath = sys.argv[2]

    # Load the specification (which kernel template shall be instantiated
    # with which template arguments) from a JSON-file.
    with open(inFilePath, "r") as inFile:
        kernelsInfo = json.load(inFile)

    with open(outFilePath, "w") as outFile:
        outFile.write("// This file was generated by {}. Don't edit manually!\n\n".format(sys.argv[0]))
        outFile.write("#include <runtime/local/context/DaphneContext.h>\n\n")
        for kernelInfo in kernelsInfo:
            kernelTemplateInfo = kernelInfo["kernelTemplate"]
            # Comment reporting the kernel name.
            outFile.write("// {}\n".format("-" * 76))
            outFile.write("// {}\n".format(kernelTemplateInfo["opName"]))
            outFile.write("// {}\n\n".format("-" * 76))
            # Include for the required header.
            outFile.write("#include <runtime/local/kernels/{}>\n\n".format(kernelTemplateInfo["header"]))
            # One function per instantiation of the kernel.
            outFile.write("extern \"C\" {\n\n")
            opCodes = kernelInfo.get("opCodes", None)
            for instantiation in kernelInfo["instantiations"]:
                generateKernelInstantiation(kernelTemplateInfo, instantiation, opCodes, outFile)
            outFile.write("}\n\n")
