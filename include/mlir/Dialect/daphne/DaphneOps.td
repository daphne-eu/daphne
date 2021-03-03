#ifndef INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNEOPS_H
#define INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNEOPS_H

include "DaphneDialect.td"
include "DaphneTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Daphne_PrintOp : Daphne_Op<"print", []> {
  let summary = "Print Operation.";

  let arguments = (ins AnyType:$input);
}

def Daphne_ConstantOp : Daphne_Op<"constant", [ConstantLike, NoSideEffect]> {
    let summary = "constant operation";

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTypeOf<[ScalarType, MatrixType]>:$result);

    let builders = [
        OpBuilderDAG<(ins "Attribute":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,
        OpBuilderDAG<(ins "double":$value), [{
            build($_builder, $_state, $_builder.getF64FloatAttr(value));
        }]>,
        //OpBuilderDAG<(ins "float":$value), [{
        //    build($_builder, $_state, $_builder.getF32FloatAttr(value));
        //}]>,
        OpBuilderDAG<(ins "int64_t":$value), [{
            build($_builder, $_state, $_builder.getIntegerAttr($_builder.getIntegerType(64, true), value));
        }]>,
        //OpBuilderDAG<(ins "int":$value), [{
        //    build($_builder, $_state, $_builder.getI32IntegerAttr(value));
        //}]>,
    ];

    let hasFolder = 1;
}

def Daphne_ReturnOp : Daphne_Op<"return", [NoSideEffect, Terminator, ReturnLike, HasParent<"FuncOp">]> {
    let summary = "return operation";

    let arguments = (ins Variadic<AnyType>:$operands);

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let builders = [
        OpBuilderDAG<(ins), [{
            build($_builder, $_state, llvm::None);
        }]>
    ];
}

def Daphne_CallKernelOp : Daphne_Op<"call_kernel"> {
    let summary = "kernel operation for C";

    let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>);

    let builders = [
        OpBuilderDAG<(ins "StringAttr":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
        }]>,
        OpBuilderDAG<(ins "StringRef":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            build($_builder, $_state, $_builder.getStringAttr(callee), operands, results);
        }]>,
    ];

    let extraClassDeclaration = [{
        StringAttr getCalleeAttr() { return (*this)->getAttrOfType<StringAttr>("callee"); }
    }];
}

class Daphne_ElementwiseBinaryOp<string mnemonic, list<OpTrait> traits = []> :
        Daphne_Op<mnemonic, !listconcat(traits, [NoSideEffect, TypesMatchWith<"left and right operand must be of the same type", "lhs", "rhs", "$_self">])> {
    let arguments = (ins AnyTypeOf<[ScalarType, MatrixType]>:$lhs, AnyTypeOf<[ScalarType, MatrixType]>:$rhs);
    let results = (outs AnyTypeOf<[ScalarType, MatrixType]>:$result);

    let builders = [
        OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs), [{
            $_state.addTypes(lhs.getType());
            $_state.addOperands({lhs, rhs});
        }]>
    ];
}

def Daphne_AddOp : Daphne_ElementwiseBinaryOp<"add", [SameOperandsAndResultShape]> {
    let summary = "add operation";
}

def Daphne_SubOp : Daphne_ElementwiseBinaryOp<"sub", [SameOperandsAndResultShape]> {
    let summary = "subtraction operation";
}

def Daphne_MulOp : Daphne_ElementwiseBinaryOp<"mul", [SameOperandsAndResultShape]> {
    let summary = "mul operation";
}

def Daphne_RandOp : Daphne_Op<"rand", [TypesMatchWith<"min and max arguments must be of the same type", "min", "max", "$_self">]> {
    let summary = "rand operation";

    let arguments = (ins AnyInteger:$rows, AnyInteger:$cols,
        AnyInteger:$seed, AnyFloat:$sparsity,
        ScalarType:$min, ScalarType:$max);
    let results = (outs MatrixType:$output);
}

def Daphne_TransposeOp : Daphne_Op<"transpose", [NoSideEffect]> {
    let summary = "transpose operation";

    let arguments = (ins MatrixType:$input);
    let results = (outs MatrixType:$result);

    let builders = [
        OpBuilderDAG<(ins "Value":$input), [{
            auto ty = input.getType().cast<MatrixType>();
            build($_builder, $_state, ty, input);
        }]>,
    ];
}

def Daphne_SetCellOp : Daphne_Op<"setCell",  []> {
    let summary = "set cell operation";

    let arguments = (ins MatrixType:$mat, Index:$row, Index:$col, ScalarType:$val);
}

#endif //INCLUDE_MLIR_DIALECT_DAPHNE_DAPHNEOPS_H
