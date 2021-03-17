#ifndef DAPHNEC_BUILTINS_H
#define DAPHNEC_BUILTINS_H

#include "mlir/Dialect/daphne/Daphne.h"

#include "antlr4-runtime.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

#include <vector>
#include <string>

template<typename T>
struct Builtin
{
    std::vector<unsigned int> expectedNumOfParams;

    Builtin(std::vector<unsigned int> expectedNumOfParams);
    virtual ~Builtin();

    mlir::LogicalResult checkNumParams(mlir::Location &loc, llvm::StringRef name, size_t size);
    virtual T create(mlir::OpBuilder builder,
                     mlir::Location &loc,
                     mlir::ValueRange values) = 0;
};

struct PrintBuiltin : public Builtin<mlir::daphne::PrintOp>
{
    using Builtin<mlir::daphne::PrintOp>::Builtin;
    static const llvm::StringRef name;

    PrintBuiltin() : Builtin({1})
    {
    };
    mlir::daphne::PrintOp create(mlir::OpBuilder builder,
                                 mlir::Location &loc,
                                 mlir::ValueRange values) override;
};

struct RandBuiltin : public Builtin<mlir::daphne::RandOp>
{
    using Builtin<mlir::daphne::RandOp>::Builtin;
    static const llvm::StringRef name;

    RandBuiltin() : Builtin({2, 4})
    {
    };
    mlir::daphne::RandOp create(mlir::OpBuilder builder,
                                mlir::Location &loc,
                                mlir::ValueRange values) override;
};

struct TransposeBuiltin : public Builtin<mlir::daphne::TransposeOp>
{
    using Builtin<mlir::daphne::TransposeOp>::Builtin;
    static const llvm::StringRef name;

    TransposeBuiltin() : Builtin({1})
    {
    };
    mlir::daphne::TransposeOp create(mlir::OpBuilder builder,
                                     mlir::Location &loc,
                                     mlir::ValueRange values) override;
};

struct Builtins
{
    static antlrcpp::Any build(mlir::OpBuilder &builder,
                               mlir::Location &loc,
                               mlir::ValueRange values,
                               const std::string &name);
};

#endif //DAPHNEC_BUILTINS_H
