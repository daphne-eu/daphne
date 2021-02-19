#include "Builtins.h"
#include "DaphneParser.h"
#include "MLIRGenVisitors.h"
#include "mlir/Dialect/daphne/Daphne.h"

#include "antlr4-runtime.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"

#include <regex>

using namespace mlir;
using namespace mlir::daphne;
using namespace mlir_gen;

using daphne_antlr::DaphneParser;
using daphne_antlr::DaphneLexer;
using daphne_antlr::DaphneParserBaseVisitor;
using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallDenseMap;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

DaphneMlirVisitor::DaphneMlirVisitor(OpBuilder &builder) : builder(builder)
{
}

Location DaphneMlirVisitor::getLocMLIR(antlr4::Token *token)
{
    auto filename = token->getTokenSource()->getSourceName();
    auto line = token->getLine();
    auto col = token->getCharPositionInLine();
    return builder.getFileLineColLoc(builder.getIdentifier(filename), line, col);
}

antlrcpp::Any DaphneMlirVisitor::visitFloatType(DaphneParser::FloatTypeContext *ctx)
{
    switch (ctx->FLOAT_TYPE()->getSymbol()->getType()) {
    case DaphneLexer::F64: return builder.getF64Type();
    case DaphneLexer::F32: return builder.getF32Type();
    }
    llvm_unreachable("Parser does not handle float type bit-size");
}

antlrcpp::Any DaphneMlirVisitor::visitIntegerType(DaphneParser::IntegerTypeContext *ctx)
{
    switch (ctx->INTEGER_TYPE()->getSymbol()->getType()) {
    case DaphneLexer::I64: return builder.getI64Type();
    case DaphneLexer::I32: return builder.getI32Type();
    }
    llvm_unreachable("Parser does not handle integer type bit-size");
}

std::string removeLiteralUnderscores(llvm::StringRef literal)
{
    auto str = literal.str();
    str.erase(std::remove(str.begin(), str.end(), '_'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '+'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '-'), str.end());
    return str;
}

llvm::Optional<llvm::APFloat> DaphneMlirVisitor::parseFloatLiteral(Location loc, llvm::StringRef floatLiteral)
{
    return llvm::APFloat(std::stod(removeLiteralUnderscores(floatLiteral)));
}

llvm::Optional<llvm::APInt> DaphneMlirVisitor::parseIntegerLiteral(Location loc, llvm::StringRef decimalLiteral,
                                                                   unsigned int bitWidth)
{
    auto literal = removeLiteralUnderscores(decimalLiteral);
    llvm::APInt number;
    if (llvm::StringRef(literal).getAsInteger(10, number)) {
        emitError(loc) << "Unable to parse integer literal\n";
        return llvm::None;
    }
    if (number.getBitWidth() > bitWidth) {
        emitError(loc) << "Integer literal does not fit in variable, too small bit width. Needed: `"
                << number.getBitWidth()
                << "` Actual: `" << bitWidth << "`\n";
        return llvm::None;
    }
    else if (number.getBitWidth() < bitWidth) {
        number = number.zext(bitWidth);
    }
    return std::move(number);
}

std::string DaphneMlirVisitor::parseStringLiteral(llvm::StringRef rawStringLiteral)
{
    // remove starting and ending "
    std::string str = std::string(rawStringLiteral.substr(1, rawStringLiteral.size() - 2));

    //std::string notBackslash = R"([^\\])";
    //auto multiple2Backslash = R"((\\)*)";
    //auto backslash = "(" + notBackslash + multiple2Backslash + R"()?\\)";
    // FIXME: "\\n" should lead to "\n" (non new-line)
    str = std::regex_replace(str, std::regex(R"(\\n)"), "\n");
    str = std::regex_replace(str, std::regex(R"(\\t)"), "\t");
    str = std::regex_replace(str, std::regex(R"(\\b)"), "\b");
    str = std::regex_replace(str, std::regex(R"(\\f)"), "\f");
    str = std::regex_replace(str, std::regex(R"(\\r)"), "\r");
    str = std::regex_replace(str, std::regex(R"(\\')"), "\'");
    str = std::regex_replace(str, std::regex(R"(\\")"), "\"");
    return str;
}

antlrcpp::Any FileVisitor::visitFile(DaphneParser::FileContext *ctx)
{
    auto module = ModuleOp::create(getLocMLIR(ctx->start));

    auto *body = module.getBody();
    builder.setInsertionPoint(body, body->begin());

    for (auto item : ctx->item()) {
        ItemVisitor itemVisitor(builder);
        auto op = itemVisitor.visitItem(item);
        if (op.isNull()) {
            // item could not be created
            return nullptr;
        }
        // module.push_back(op);
    }
    return module;
}

antlrcpp::Any ItemVisitor::visitItem(DaphneParser::ItemContext *ctx)
{
    FunctionVisitor functionVisitor(builder);
    return functionVisitor.visitFunction(ctx->function());
}

antlrcpp::Any FunctionVisitor::visitFunction(DaphneParser::FunctionContext *ctx)
{
    auto loc = getLocMLIR(ctx->KW_DEF()->getSymbol());
    auto *funcBlock = new Block();
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(funcBlock, funcBlock->begin());

        if (ctx->functionArgs() != nullptr && failed(visitFunctionArgs(ctx->functionArgs()))) {
            return nullptr;
        }

        auto endLoc = getLocMLIR(ctx->RPAREN()->getSymbol());
        if (failed(visitBlockStatement(ctx->blockStatement()))) {
            return nullptr;
        }
        if (!funcBlock->back().mightHaveTrait<OpTrait::IsTerminator>()) {
            // implicitly return last operation
            builder.create<daphne::ReturnOp>(endLoc);
        }
        if (funcBlock->empty()) {
            builder.create<daphne::ReturnOp>(endLoc);
        }
    }

    auto *terminator = funcBlock->getTerminator();
    auto funcType = FunctionType::get(builder.getContext(), funcBlock->getArgumentTypes(), terminator->getOperandTypes());
    // TODO The function name prefix should probably not be inserted here.
    auto func = builder.create<FuncOp>(loc, "_mlir__mlir_ciface_" + ctx->IDENTIFIER()->getText(), funcType);
    func.push_back(funcBlock);
    return func.getOperation();
}

antlrcpp::Any FunctionVisitor::visitFunctionArgs(DaphneParser::FunctionArgsContext *ctx)
{
    for (auto arg : ctx->functionArg()) {
        if (failed(visitFunctionArg(arg))) {
            return failure();
        }
    }
    return success();
}

antlrcpp::Any FunctionVisitor::visitFunctionArg(DaphneParser::FunctionArgContext *ctx)
{
    Type argType = visit(ctx->type());
    auto arg = builder.getBlock()->addArgument(argType);
    if (failed(declareVar(ctx->IDENTIFIER()->getText(), arg))) {
        // NOTE: consider continuing parsing for further errors
        return failure();
    }
    return success();
}

antlrcpp::Any FunctionVisitor::visitLiteralExpression(DaphneParser::LiteralExpressionContext *ctx)
{
    auto res = visitLiteralExpressionRule(ctx->literalExpressionRule());
    auto loc = getLocMLIR(ctx->start);
    Value value = nullptr;
    if (res.is<APFloat>()) {
        // TODO: get bit type, if specified
        Type type = builder.getF64Type();
        value = builder.create<daphne::ConstantOp>(loc,
                builder.getFloatAttr(type, res.as<APFloat>()));
    }
    else if (res.is<APInt>()) {
        // TODO: get bit type, if specified
        Type type = builder.getIntegerType(64, true);
        value = builder.create<daphne::ConstantOp>(loc,
                builder.getIntegerAttr(type, res.as<APInt>()));
    }
    if (value == nullptr)
        emitError(loc) << "failed to create literal expression";
    return value;
}

antlrcpp::Any FunctionVisitor::visitLiteralExpressionRule(DaphneParser::LiteralExpressionRuleContext *ctx)
{
    if (auto floatLiteral = ctx->FLOAT_LITERAL()) {
        auto loc = getLocMLIR(floatLiteral->getSymbol());
        if (auto val = parseFloatLiteral(loc, floatLiteral->getText()))
            return static_cast<APFloat> (val.getValue());
    }
    else if (auto intLiteral = ctx->INTEGER_LITERAL()) {
        // TODO: get bit type, if specified
        Type type = builder.getI64Type();
        auto bitWidth = type.cast<IntegerType>().getWidth();
        auto loc = getLocMLIR(intLiteral->getSymbol());
        if (auto val = parseIntegerLiteral(loc, intLiteral->getText(), bitWidth))
            return static_cast<APInt> (val.getValue());
    }
    else if (auto strLiteral = ctx->STRING_LITERAL()) {
        return strLiteral->getText();
    }
    else if (auto matrixLiteralCtx = ctx->matrixLiteral()) {
        return visitMatrixLiteral(matrixLiteralCtx);
    }
    else {
        llvm_unreachable("Literal not handled!");
    }
    emitError(getLocMLIR(ctx->start)) << "Reading literal failed";
    return nullptr;
}

antlrcpp::Any FunctionVisitor::visitMatrixLiteral(DaphneParser::MatrixLiteralContext *ctx)
{
    return visitMatrixLiteralElements(ctx->matrixLiteralElements());
}

antlrcpp::Any FunctionVisitor::visitMatrixLiteralElements(DaphneParser::MatrixLiteralElementsContext *ctx)
{
    MatrixLiteral matrixLiteral;
    for (auto litExprRule : ctx->literalExpressionRule()) {
        auto loc = getLocMLIR(litExprRule->start);
        auto val = visitLiteralExpressionRule(litExprRule);
        if (val.isNull()) {
            emitError(loc) << "Could not parse at least one of the literal expressions";
            return nullptr;
        }
        matrixLiteral.addData(loc, builder, std::move(val));
    }
    return matrixLiteral;
}

antlrcpp::Any FunctionVisitor::visitBlockStatement(DaphneParser::BlockStatementContext *ctx)
{
    // scope handling here?
    for (auto statement : ctx->statement()) {
        auto result = visitStatement(statement);
        if (result.isNull() || (result.is<LogicalResult>() && failed(result))) {
            return failure();
        }
    }
    return success();
}

antlrcpp::Any FunctionVisitor::visitExpressionStatement(DaphneParser::ExpressionStatementContext *ctx)
{
    if (visit(ctx->expression()).isNotNull()) {
        return success();
    }
    return failure();
}

antlrcpp::Any FunctionVisitor::visitAssignmentExpression(DaphneParser::AssignmentExpressionContext *ctx)
{
    auto varName = ctx->IDENTIFIER()->getText();
    // TODO: check if shadowing is allowed
    auto val = visit(ctx->expression());
    if (val.isNull()) {
        return nullptr;
    }
    else if (val.is<Operation *>()) {
        emitError(getLocMLIR(ctx->EQ()->getSymbol())) << "Can't assign expression that does not return a value";
        return nullptr;
    }
    if (failed(declareVar(varName, val, true))) {
        return nullptr;
    }
    return val;
}

antlrcpp::Any FunctionVisitor::visitGroupedExpression(DaphneParser::GroupedExpressionContext *ctx)
{
    return visit(ctx->expression());
}

antlrcpp::Any FunctionVisitor::visitLetStatement(DaphneParser::LetStatementContext *ctx)
{
    auto val = visit(ctx->expression());
    if (val.isNull()) {
        return failure();
    }
    if (val.is<Operation *>()) {
        emitError(getLocMLIR(ctx->start)) << "Can't initialize variable with expression that does not return a value";
        return failure();
    }
    if (failed(declareVar(ctx->IDENTIFIER()->getText(), val))) {
        return failure();
    }
    return success();
}

antlrcpp::Any FunctionVisitor::visitIdentifierExpression(DaphneParser::IdentifierExpressionContext *ctx)
{
    auto varName = ctx->IDENTIFIER()->getText();
    auto it = symbolTable.find(varName);
    if (it != symbolTable.end()) {
        return it->second;
    }
    else {
        emitError(getLocMLIR(ctx->IDENTIFIER()->getSymbol())) << "Variable `" << varName
                << "` is not defined in this scope";
    }
    return nullptr;
}

antlrcpp::Any FunctionVisitor::visitCallExpression(DaphneParser::CallExpressionContext *ctx)
{
    std::vector<Value> parameters;
    if (ctx->parameters()) {
        auto parametersRet = visitParameters(ctx->parameters());
        if (parametersRet.isNull()) {
            return nullptr;
        }
        if (!parametersRet.is<std::vector < Value >> ()) {
            llvm_unreachable("`visitParameters()` returned wrong type!");
        }
        parameters = parametersRet.as<std::vector < Value >> ();
    }
    auto loc = getLocMLIR(ctx->IDENTIFIER()->getSymbol());
    auto builtin = Builtins::build(builder, loc, parameters, ctx->fn->getText());
    if (builtin.isNull()) {
        // TODO: check user defined functions
        emitError(getLocMLIR(ctx->start)) << "Unknown function call `" << ctx->getText() << '`';
        return nullptr;
    }
    if (builtin.is<Value>()) {
        return builtin.as<Value>();
    }
    return builtin.as<Operation *>();
}

antlrcpp::Any FunctionVisitor::visitArithmeticExpression(DaphneParser::ArithmeticExpressionContext *ctx)
{
    auto lhsAny = visit(ctx->lhs);
    if (lhsAny.isNull()) {
        emitError(getLocMLIR(ctx->lhs->start)) << "left hand side of arithmetic expression did not return a value";
        return nullptr;
    }
    Value lhs = lhsAny;
    auto rhsAny = visit(ctx->rhs);
    if (rhsAny.isNull()) {
        emitError(getLocMLIR(ctx->rhs->start)) << "right hand side of arithmetic expression did not return a value";
        return nullptr;
    }
    Value rhs = rhsAny;
    auto loc = getLocMLIR(ctx->op);
    Value retVal = nullptr;
    //  if (ctx->AT()) // @
    //    retVal = builder.create<MatMulOp>(loc, lhs, rhs);
    if (ctx->STAR()) // *
        retVal = builder.create<MulOp>(loc, lhs, rhs);
    //  if (ctx->SLASH()) // /
    //    retVal = builder.create<DivOp>(loc, lhs, rhs);
    if (ctx->PLUS()) // +
        retVal = builder.create<AddOp>(loc, lhs, rhs);
    if (ctx->MINUS()) // -
        retVal = builder.create<SubOp>(loc, lhs, rhs);
    if (retVal == nullptr)
        llvm_unreachable("Arithmetic expression operation not implemented");
    return retVal;
}

antlrcpp::Any FunctionVisitor::visitParameters(DaphneParser::ParametersContext *ctx)
{
    std::vector<Value> parameters;
    for (auto parameter : ctx->parameter()) {
        auto val = visitParameter(parameter);
        if (val.isNull()) {
            return nullptr;
        }
        parameters.push_back(val);
    }
    return parameters;
}

antlrcpp::Any FunctionVisitor::visitParameter(DaphneParser::ParameterContext *ctx)
{
    auto valRet = visit(ctx->expression());
    if (valRet.isNull()) {
        return nullptr;
    }
    else if (valRet.is<Operation *>()) {
        emitError(getLocMLIR(ctx->start)) << "Parameter expression has to return a value";
        return nullptr;
    }
    return valRet;
}

antlrcpp::Any FunctionVisitor::visitStatement(DaphneParser::StatementContext *ctx)
{
    if (ctx->blockStatement()) {
        return visitBlockStatement(ctx->blockStatement());
    }
    else if (ctx->expressionStatement()) {
        return visitExpressionStatement(ctx->expressionStatement());
    }
    else if (ctx->letStatement()) {
        return visitLetStatement(ctx->letStatement());
    }
    emitError(getLocMLIR(ctx->start)) << "Statement kind is not handled\n";
    return failure();
}

LogicalResult FunctionVisitor::declareVar(std::string name, Value value, bool allowShadowing)
{
    if (symbolTable.count(name) && !allowShadowing) {
        emitError(value.getLoc()) << "Variable with name `" << name << "` already exists and shadowing is not allowed here";
        return failure();
    }
    symbolTable[name] = value;
    return success();
}

void MatrixLiteral::addData(Location &loc, OpBuilder &builder, antlrcpp::Any data)
{
    if (data.is<MatrixLiteral>()) {
        auto childLit = data.as<MatrixLiteral>();
        if (!initialized) {
            initialized = true;
            rows = 1;
            elementType = childLit.elementType;
            cols = childLit.cols;
        }
        else {
            rows++;
        }

        if (childLit.rows != -1) {
            emitError(loc) << "Matrix literal is nested too often (only exactly 2 nestings are supported)";
        }
        else if (elementType != childLit.elementType) {
            emitError(loc) << "Matrix literal element types don't match";
            return;
        }
        else if (cols != childLit.cols) {
            emitError(loc) << "Number of elements in every column have to match for matrix literals";
        }
        if (elementType.isa<FloatType>()) {
            linearizedFloatData.insert(linearizedFloatData.end(),
                                       childLit.linearizedFloatData.begin(),
                                       childLit.linearizedFloatData.end());
        }
        else {
            linearizedIntData.insert(linearizedIntData.end(),
                                     childLit.linearizedIntData.begin(),
                                     childLit.linearizedIntData.end());
        }
    }
    else if (rows != -1) {
        emitError(loc) << "Matrix literals have to be nested exactly 2 times";
        emitRemark(loc)
                << "Single 'array'/'row' which can be reshaped to a matrix shape will be supported in the future:\n"
                << "Some syntax similar to the following line will be supported.\n" << "\tlet a: 2x2 = [1., 2., 3., 4.];";
        return;
    }
    else if (data.is<APFloat>()) {
        if (!initialized) {
            initialized = true;
            // TODO: different bit-widths
            elementType = builder.getF64Type();
            cols = 1;
        }
        else {
            cols++;
        }
        linearizedFloatData.push_back(data.as<APFloat>());
    }
    else {
        if (!initialized) {
            initialized = true;
            // TODO: different bit-widths
            elementType = builder.getI64Type();
            cols = 1;
        }
        else {
            cols++;
        }
        linearizedIntData.push_back(data.as<APInt>());

    }
}
