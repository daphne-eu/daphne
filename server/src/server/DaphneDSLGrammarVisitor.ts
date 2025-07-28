// Generated from ./server/DaphneDSLGrammar.g4 by ANTLR 4.9.0-SNAPSHOT


import { ParseTreeVisitor } from "antlr4ts/tree/ParseTreeVisitor";

import { LiteralExprContext } from "./DaphneDSLGrammarParser";
import { ArgExprContext } from "./DaphneDSLGrammarParser";
import { IdentifierExprContext } from "./DaphneDSLGrammarParser";
import { ParanthesesExprContext } from "./DaphneDSLGrammarParser";
import { CallExprContext } from "./DaphneDSLGrammarParser";
import { CastExprContext } from "./DaphneDSLGrammarParser";
import { RightIdxFilterExprContext } from "./DaphneDSLGrammarParser";
import { RightIdxExtractExprContext } from "./DaphneDSLGrammarParser";
import { MinusExprContext } from "./DaphneDSLGrammarParser";
import { MatmulExprContext } from "./DaphneDSLGrammarParser";
import { PowExprContext } from "./DaphneDSLGrammarParser";
import { ModExprContext } from "./DaphneDSLGrammarParser";
import { MulExprContext } from "./DaphneDSLGrammarParser";
import { AddExprContext } from "./DaphneDSLGrammarParser";
import { CmpExprContext } from "./DaphneDSLGrammarParser";
import { ConjExprContext } from "./DaphneDSLGrammarParser";
import { DisjExprContext } from "./DaphneDSLGrammarParser";
import { CondExprContext } from "./DaphneDSLGrammarParser";
import { MatrixLiteralExprContext } from "./DaphneDSLGrammarParser";
import { ColMajorFrameLiteralExprContext } from "./DaphneDSLGrammarParser";
import { RowMajorFrameLiteralExprContext } from "./DaphneDSLGrammarParser";
import { ScriptContext } from "./DaphneDSLGrammarParser";
import { StatementContext } from "./DaphneDSLGrammarParser";
import { ImportStatementContext } from "./DaphneDSLGrammarParser";
import { BlockStatementContext } from "./DaphneDSLGrammarParser";
import { ExprStatementContext } from "./DaphneDSLGrammarParser";
import { AssignStatementContext } from "./DaphneDSLGrammarParser";
import { IfStatementContext } from "./DaphneDSLGrammarParser";
import { WhileStatementContext } from "./DaphneDSLGrammarParser";
import { ForStatementContext } from "./DaphneDSLGrammarParser";
import { FunctionStatementContext } from "./DaphneDSLGrammarParser";
import { ReturnStatementContext } from "./DaphneDSLGrammarParser";
import { FunctionArgsContext } from "./DaphneDSLGrammarParser";
import { FunctionArgContext } from "./DaphneDSLGrammarParser";
import { FunctionRetTypesContext } from "./DaphneDSLGrammarParser";
import { FuncTypeDefContext } from "./DaphneDSLGrammarParser";
import { ExprContext } from "./DaphneDSLGrammarParser";
import { FrameRowContext } from "./DaphneDSLGrammarParser";
import { IndexingContext } from "./DaphneDSLGrammarParser";
import { RangeContext } from "./DaphneDSLGrammarParser";
import { LiteralContext } from "./DaphneDSLGrammarParser";
import { BoolLiteralContext } from "./DaphneDSLGrammarParser";


/**
 * This interface defines a complete generic visitor for a parse tree produced
 * by `DaphneDSLGrammarParser`.
 *
 * @param <Result> The return type of the visit operation. Use `void` for
 * operations with no return type.
 */
export interface DaphneDSLGrammarVisitor<Result> extends ParseTreeVisitor<Result> {
	/**
	 * Visit a parse tree produced by the `literalExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitLiteralExpr?: (ctx: LiteralExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `argExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitArgExpr?: (ctx: ArgExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `identifierExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitIdentifierExpr?: (ctx: IdentifierExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `paranthesesExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitParanthesesExpr?: (ctx: ParanthesesExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `callExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitCallExpr?: (ctx: CallExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `castExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitCastExpr?: (ctx: CastExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `rightIdxFilterExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitRightIdxFilterExpr?: (ctx: RightIdxFilterExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `rightIdxExtractExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitRightIdxExtractExpr?: (ctx: RightIdxExtractExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `minusExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitMinusExpr?: (ctx: MinusExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `matmulExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitMatmulExpr?: (ctx: MatmulExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `powExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitPowExpr?: (ctx: PowExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `modExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitModExpr?: (ctx: ModExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `mulExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitMulExpr?: (ctx: MulExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `addExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitAddExpr?: (ctx: AddExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `cmpExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitCmpExpr?: (ctx: CmpExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `conjExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitConjExpr?: (ctx: ConjExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `disjExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitDisjExpr?: (ctx: DisjExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `condExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitCondExpr?: (ctx: CondExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `matrixLiteralExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitMatrixLiteralExpr?: (ctx: MatrixLiteralExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `colMajorFrameLiteralExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitColMajorFrameLiteralExpr?: (ctx: ColMajorFrameLiteralExprContext) => Result;

	/**
	 * Visit a parse tree produced by the `rowMajorFrameLiteralExpr`
	 * labeled alternative in `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitRowMajorFrameLiteralExpr?: (ctx: RowMajorFrameLiteralExprContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.script`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitScript?: (ctx: ScriptContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.statement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitStatement?: (ctx: StatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.importStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitImportStatement?: (ctx: ImportStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.blockStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitBlockStatement?: (ctx: BlockStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.exprStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitExprStatement?: (ctx: ExprStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.assignStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitAssignStatement?: (ctx: AssignStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.ifStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitIfStatement?: (ctx: IfStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.whileStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitWhileStatement?: (ctx: WhileStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.forStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitForStatement?: (ctx: ForStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.functionStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitFunctionStatement?: (ctx: FunctionStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.returnStatement`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitReturnStatement?: (ctx: ReturnStatementContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.functionArgs`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitFunctionArgs?: (ctx: FunctionArgsContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.functionArg`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitFunctionArg?: (ctx: FunctionArgContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.functionRetTypes`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitFunctionRetTypes?: (ctx: FunctionRetTypesContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.funcTypeDef`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitFuncTypeDef?: (ctx: FuncTypeDefContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.expr`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitExpr?: (ctx: ExprContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.frameRow`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitFrameRow?: (ctx: FrameRowContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.indexing`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitIndexing?: (ctx: IndexingContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.range`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitRange?: (ctx: RangeContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.literal`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitLiteral?: (ctx: LiteralContext) => Result;

	/**
	 * Visit a parse tree produced by `DaphneDSLGrammarParser.boolLiteral`.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	visitBoolLiteral?: (ctx: BoolLiteralContext) => Result;
}

