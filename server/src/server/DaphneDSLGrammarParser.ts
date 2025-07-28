// Generated from ./server/DaphneDSLGrammar.g4 by ANTLR 4.9.0-SNAPSHOT


import { ATN } from "antlr4ts/atn/ATN";
import { ATNDeserializer } from "antlr4ts/atn/ATNDeserializer";
import { FailedPredicateException } from "antlr4ts/FailedPredicateException";
import { NotNull } from "antlr4ts/Decorators";
import { NoViableAltException } from "antlr4ts/NoViableAltException";
import { Override } from "antlr4ts/Decorators";
import { Parser } from "antlr4ts/Parser";
import { ParserRuleContext } from "antlr4ts/ParserRuleContext";
import { ParserATNSimulator } from "antlr4ts/atn/ParserATNSimulator";
import { ParseTreeListener } from "antlr4ts/tree/ParseTreeListener";
import { ParseTreeVisitor } from "antlr4ts/tree/ParseTreeVisitor";
import { RecognitionException } from "antlr4ts/RecognitionException";
import { RuleContext } from "antlr4ts/RuleContext";
//import { RuleVersion } from "antlr4ts/RuleVersion";
import { TerminalNode } from "antlr4ts/tree/TerminalNode";
import { Token } from "antlr4ts/Token";
import { TokenStream } from "antlr4ts/TokenStream";
import { Vocabulary } from "antlr4ts/Vocabulary";
import { VocabularyImpl } from "antlr4ts/VocabularyImpl";

import * as Utils from "antlr4ts/misc/Utils";

import { DaphneDSLGrammarVisitor } from "./DaphneDSLGrammarVisitor";


export class DaphneDSLGrammarParser extends Parser {
	public static readonly T__0 = 1;
	public static readonly T__1 = 2;
	public static readonly T__2 = 3;
	public static readonly T__3 = 4;
	public static readonly T__4 = 5;
	public static readonly T__5 = 6;
	public static readonly T__6 = 7;
	public static readonly T__7 = 8;
	public static readonly T__8 = 9;
	public static readonly T__9 = 10;
	public static readonly T__10 = 11;
	public static readonly T__11 = 12;
	public static readonly T__12 = 13;
	public static readonly T__13 = 14;
	public static readonly T__14 = 15;
	public static readonly T__15 = 16;
	public static readonly T__16 = 17;
	public static readonly T__17 = 18;
	public static readonly T__18 = 19;
	public static readonly T__19 = 20;
	public static readonly T__20 = 21;
	public static readonly T__21 = 22;
	public static readonly T__22 = 23;
	public static readonly T__23 = 24;
	public static readonly T__24 = 25;
	public static readonly T__25 = 26;
	public static readonly T__26 = 27;
	public static readonly T__27 = 28;
	public static readonly T__28 = 29;
	public static readonly T__29 = 30;
	public static readonly T__30 = 31;
	public static readonly T__31 = 32;
	public static readonly KW_IF = 33;
	public static readonly KW_ELSE = 34;
	public static readonly KW_WHILE = 35;
	public static readonly KW_DO = 36;
	public static readonly KW_FOR = 37;
	public static readonly KW_IN = 38;
	public static readonly KW_TRUE = 39;
	public static readonly KW_FALSE = 40;
	public static readonly KW_AS = 41;
	public static readonly KW_DEF = 42;
	public static readonly KW_RETURN = 43;
	public static readonly KW_IMPORT = 44;
	public static readonly DATA_TYPE = 45;
	public static readonly VALUE_TYPE = 46;
	public static readonly INT_LITERAL = 47;
	public static readonly FLOAT_LITERAL = 48;
	public static readonly STRING_LITERAL = 49;
	public static readonly IDENTIFIER = 50;
	public static readonly SCRIPT_STYLE_LINE_COMMENT = 51;
	public static readonly C_STYLE_LINE_COMMENT = 52;
	public static readonly MULTILINE_BLOCK_COMMENT = 53;
	public static readonly WS = 54;
	public static readonly RULE_script = 0;
	public static readonly RULE_statement = 1;
	public static readonly RULE_importStatement = 2;
	public static readonly RULE_blockStatement = 3;
	public static readonly RULE_exprStatement = 4;
	public static readonly RULE_assignStatement = 5;
	public static readonly RULE_ifStatement = 6;
	public static readonly RULE_whileStatement = 7;
	public static readonly RULE_forStatement = 8;
	public static readonly RULE_functionStatement = 9;
	public static readonly RULE_returnStatement = 10;
	public static readonly RULE_functionArgs = 11;
	public static readonly RULE_functionArg = 12;
	public static readonly RULE_functionRetTypes = 13;
	public static readonly RULE_funcTypeDef = 14;
	public static readonly RULE_expr = 15;
	public static readonly RULE_frameRow = 16;
	public static readonly RULE_indexing = 17;
	public static readonly RULE_range = 18;
	public static readonly RULE_literal = 19;
	public static readonly RULE_boolLiteral = 20;
	// tslint:disable:no-trailing-whitespace
	public static readonly ruleNames: string[] = [
		"script", "statement", "importStatement", "blockStatement", "exprStatement", 
		"assignStatement", "ifStatement", "whileStatement", "forStatement", "functionStatement", 
		"returnStatement", "functionArgs", "functionArg", "functionRetTypes", 
		"funcTypeDef", "expr", "frameRow", "indexing", "range", "literal", "boolLiteral",
	];

	private static readonly _LITERAL_NAMES: Array<string | undefined> = [
		undefined, "';'", "'{'", "'}'", "'.'", "','", "'='", "'('", "')'", "':'", 
		"'->'", "'<'", "'>'", "'$'", "'::'", "'[['", "']]'", "'+'", "'-'", "'@'", 
		"'^'", "'%'", "'*'", "'/'", "'=='", "'!='", "'<='", "'>='", "'&&'", "'||'", 
		"'?'", "'['", "']'", "'if'", "'else'", "'while'", "'do'", "'for'", "'in'", 
		"'true'", "'false'", "'as'", "'def'", "'return'", "'import'",
	];
	private static readonly _SYMBOLIC_NAMES: Array<string | undefined> = [
		undefined, undefined, undefined, undefined, undefined, undefined, undefined, 
		undefined, undefined, undefined, undefined, undefined, undefined, undefined, 
		undefined, undefined, undefined, undefined, undefined, undefined, undefined, 
		undefined, undefined, undefined, undefined, undefined, undefined, undefined, 
		undefined, undefined, undefined, undefined, undefined, "KW_IF", "KW_ELSE", 
		"KW_WHILE", "KW_DO", "KW_FOR", "KW_IN", "KW_TRUE", "KW_FALSE", "KW_AS", 
		"KW_DEF", "KW_RETURN", "KW_IMPORT", "DATA_TYPE", "VALUE_TYPE", "INT_LITERAL", 
		"FLOAT_LITERAL", "STRING_LITERAL", "IDENTIFIER", "SCRIPT_STYLE_LINE_COMMENT", 
		"C_STYLE_LINE_COMMENT", "MULTILINE_BLOCK_COMMENT", "WS",
	];
	public static readonly VOCABULARY: Vocabulary = new VocabularyImpl(DaphneDSLGrammarParser._LITERAL_NAMES, DaphneDSLGrammarParser._SYMBOLIC_NAMES, []);

	// @Override
	// @NotNull
	public get vocabulary(): Vocabulary {
		return DaphneDSLGrammarParser.VOCABULARY;
	}
	// tslint:enable:no-trailing-whitespace

	// @Override
	public get grammarFileName(): string { return "DaphneDSLGrammar.g4"; }

	// @Override
	public get ruleNames(): string[] { return DaphneDSLGrammarParser.ruleNames; }

	// @Override
	public get serializedATN(): string { return DaphneDSLGrammarParser._serializedATN; }

	protected createFailedPredicateException(predicate?: string, message?: string): FailedPredicateException {
		return new FailedPredicateException(this, predicate, message);
	}

	constructor(input: TokenStream) {
		super(input);
		this._interp = new ParserATNSimulator(DaphneDSLGrammarParser._ATN, this);
	}
	// @RuleVersion(0)
	public script(): ScriptContext {
		let _localctx: ScriptContext = new ScriptContext(this._ctx, this.state);
		this.enterRule(_localctx, 0, DaphneDSLGrammarParser.RULE_script);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 45;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			while ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 33)) & ~0x1F) === 0 && ((1 << (_la - 33)) & ((1 << (DaphneDSLGrammarParser.KW_IF - 33)) | (1 << (DaphneDSLGrammarParser.KW_WHILE - 33)) | (1 << (DaphneDSLGrammarParser.KW_DO - 33)) | (1 << (DaphneDSLGrammarParser.KW_FOR - 33)) | (1 << (DaphneDSLGrammarParser.KW_TRUE - 33)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 33)) | (1 << (DaphneDSLGrammarParser.KW_AS - 33)) | (1 << (DaphneDSLGrammarParser.KW_DEF - 33)) | (1 << (DaphneDSLGrammarParser.KW_RETURN - 33)) | (1 << (DaphneDSLGrammarParser.KW_IMPORT - 33)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 33)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 33)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 33)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 33)))) !== 0)) {
				{
				{
				this.state = 42;
				this.statement();
				}
				}
				this.state = 47;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
			}
			this.state = 48;
			this.match(DaphneDSLGrammarParser.EOF);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public statement(): StatementContext {
		let _localctx: StatementContext = new StatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 2, DaphneDSLGrammarParser.RULE_statement);
		try {
			this.state = 59;
			this._errHandler.sync(this);
			switch ( this.interpreter.adaptivePredict(this._input, 1, this._ctx) ) {
			case 1:
				this.enterOuterAlt(_localctx, 1);
				{
				this.state = 50;
				this.blockStatement();
				}
				break;

			case 2:
				this.enterOuterAlt(_localctx, 2);
				{
				this.state = 51;
				this.exprStatement();
				}
				break;

			case 3:
				this.enterOuterAlt(_localctx, 3);
				{
				this.state = 52;
				this.assignStatement();
				}
				break;

			case 4:
				this.enterOuterAlt(_localctx, 4);
				{
				this.state = 53;
				this.ifStatement();
				}
				break;

			case 5:
				this.enterOuterAlt(_localctx, 5);
				{
				this.state = 54;
				this.whileStatement();
				}
				break;

			case 6:
				this.enterOuterAlt(_localctx, 6);
				{
				this.state = 55;
				this.forStatement();
				}
				break;

			case 7:
				this.enterOuterAlt(_localctx, 7);
				{
				this.state = 56;
				this.functionStatement();
				}
				break;

			case 8:
				this.enterOuterAlt(_localctx, 8);
				{
				this.state = 57;
				this.returnStatement();
				}
				break;

			case 9:
				this.enterOuterAlt(_localctx, 9);
				{
				this.state = 58;
				this.importStatement();
				}
				break;
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public importStatement(): ImportStatementContext {
		let _localctx: ImportStatementContext = new ImportStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 4, DaphneDSLGrammarParser.RULE_importStatement);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 61;
			this.match(DaphneDSLGrammarParser.KW_IMPORT);
			this.state = 62;
			_localctx._filePath = this.match(DaphneDSLGrammarParser.STRING_LITERAL);
			this.state = 65;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.KW_AS) {
				{
				this.state = 63;
				this.match(DaphneDSLGrammarParser.KW_AS);
				this.state = 64;
				_localctx._alias = this.match(DaphneDSLGrammarParser.STRING_LITERAL);
				}
			}

			this.state = 67;
			this.match(DaphneDSLGrammarParser.T__0);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public blockStatement(): BlockStatementContext {
		let _localctx: BlockStatementContext = new BlockStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 6, DaphneDSLGrammarParser.RULE_blockStatement);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 69;
			this.match(DaphneDSLGrammarParser.T__1);
			this.state = 73;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			while ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 33)) & ~0x1F) === 0 && ((1 << (_la - 33)) & ((1 << (DaphneDSLGrammarParser.KW_IF - 33)) | (1 << (DaphneDSLGrammarParser.KW_WHILE - 33)) | (1 << (DaphneDSLGrammarParser.KW_DO - 33)) | (1 << (DaphneDSLGrammarParser.KW_FOR - 33)) | (1 << (DaphneDSLGrammarParser.KW_TRUE - 33)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 33)) | (1 << (DaphneDSLGrammarParser.KW_AS - 33)) | (1 << (DaphneDSLGrammarParser.KW_DEF - 33)) | (1 << (DaphneDSLGrammarParser.KW_RETURN - 33)) | (1 << (DaphneDSLGrammarParser.KW_IMPORT - 33)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 33)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 33)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 33)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 33)))) !== 0)) {
				{
				{
				this.state = 70;
				this.statement();
				}
				}
				this.state = 75;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
			}
			this.state = 76;
			this.match(DaphneDSLGrammarParser.T__2);
			this.state = 78;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.T__0) {
				{
				this.state = 77;
				this.match(DaphneDSLGrammarParser.T__0);
				}
			}

			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public exprStatement(): ExprStatementContext {
		let _localctx: ExprStatementContext = new ExprStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 8, DaphneDSLGrammarParser.RULE_exprStatement);
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 80;
			this.expr(0);
			this.state = 81;
			this.match(DaphneDSLGrammarParser.T__0);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public assignStatement(): AssignStatementContext {
		let _localctx: AssignStatementContext = new AssignStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 10, DaphneDSLGrammarParser.RULE_assignStatement);
		let _la: number;
		try {
			let _alt: number;
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 87;
			this._errHandler.sync(this);
			_alt = this.interpreter.adaptivePredict(this._input, 5, this._ctx);
			while (_alt !== 2 && _alt !== ATN.INVALID_ALT_NUMBER) {
				if (_alt === 1) {
					{
					{
					this.state = 83;
					this.match(DaphneDSLGrammarParser.IDENTIFIER);
					this.state = 84;
					this.match(DaphneDSLGrammarParser.T__3);
					}
					}
				}
				this.state = 89;
				this._errHandler.sync(this);
				_alt = this.interpreter.adaptivePredict(this._input, 5, this._ctx);
			}
			this.state = 90;
			this.match(DaphneDSLGrammarParser.IDENTIFIER);
			this.state = 92;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.T__30) {
				{
				this.state = 91;
				this.indexing();
				}
			}

			this.state = 108;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			while (_la === DaphneDSLGrammarParser.T__4) {
				{
				{
				this.state = 94;
				this.match(DaphneDSLGrammarParser.T__4);
				this.state = 99;
				this._errHandler.sync(this);
				_alt = this.interpreter.adaptivePredict(this._input, 7, this._ctx);
				while (_alt !== 2 && _alt !== ATN.INVALID_ALT_NUMBER) {
					if (_alt === 1) {
						{
						{
						this.state = 95;
						this.match(DaphneDSLGrammarParser.IDENTIFIER);
						this.state = 96;
						this.match(DaphneDSLGrammarParser.T__3);
						}
						}
					}
					this.state = 101;
					this._errHandler.sync(this);
					_alt = this.interpreter.adaptivePredict(this._input, 7, this._ctx);
				}
				this.state = 102;
				this.match(DaphneDSLGrammarParser.IDENTIFIER);
				this.state = 104;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if (_la === DaphneDSLGrammarParser.T__30) {
					{
					this.state = 103;
					this.indexing();
					}
				}

				}
				}
				this.state = 110;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
			}
			this.state = 111;
			this.match(DaphneDSLGrammarParser.T__5);
			this.state = 112;
			this.expr(0);
			this.state = 113;
			this.match(DaphneDSLGrammarParser.T__0);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public ifStatement(): IfStatementContext {
		let _localctx: IfStatementContext = new IfStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 12, DaphneDSLGrammarParser.RULE_ifStatement);
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 115;
			this.match(DaphneDSLGrammarParser.KW_IF);
			this.state = 116;
			this.match(DaphneDSLGrammarParser.T__6);
			this.state = 117;
			_localctx._cond = this.expr(0);
			this.state = 118;
			this.match(DaphneDSLGrammarParser.T__7);
			this.state = 119;
			_localctx._thenStmt = this.statement();
			this.state = 122;
			this._errHandler.sync(this);
			switch ( this.interpreter.adaptivePredict(this._input, 10, this._ctx) ) {
			case 1:
				{
				this.state = 120;
				this.match(DaphneDSLGrammarParser.KW_ELSE);
				this.state = 121;
				_localctx._elseStmt = this.statement();
				}
				break;
			}
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public whileStatement(): WhileStatementContext {
		let _localctx: WhileStatementContext = new WhileStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 14, DaphneDSLGrammarParser.RULE_whileStatement);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 139;
			this._errHandler.sync(this);
			switch (this._input.LA(1)) {
			case DaphneDSLGrammarParser.KW_WHILE:
				{
				this.state = 124;
				this.match(DaphneDSLGrammarParser.KW_WHILE);
				this.state = 125;
				this.match(DaphneDSLGrammarParser.T__6);
				this.state = 126;
				_localctx._cond = this.expr(0);
				this.state = 127;
				this.match(DaphneDSLGrammarParser.T__7);
				this.state = 128;
				_localctx._bodyStmt = this.statement();
				}
				break;
			case DaphneDSLGrammarParser.KW_DO:
				{
				this.state = 130;
				this.match(DaphneDSLGrammarParser.KW_DO);
				this.state = 131;
				_localctx._bodyStmt = this.statement();
				this.state = 132;
				this.match(DaphneDSLGrammarParser.KW_WHILE);
				this.state = 133;
				this.match(DaphneDSLGrammarParser.T__6);
				this.state = 134;
				_localctx._cond = this.expr(0);
				this.state = 135;
				this.match(DaphneDSLGrammarParser.T__7);
				this.state = 137;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if (_la === DaphneDSLGrammarParser.T__0) {
					{
					this.state = 136;
					this.match(DaphneDSLGrammarParser.T__0);
					}
				}

				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public forStatement(): ForStatementContext {
		let _localctx: ForStatementContext = new ForStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 16, DaphneDSLGrammarParser.RULE_forStatement);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 141;
			this.match(DaphneDSLGrammarParser.KW_FOR);
			this.state = 142;
			this.match(DaphneDSLGrammarParser.T__6);
			this.state = 143;
			_localctx._var = this.match(DaphneDSLGrammarParser.IDENTIFIER);
			this.state = 144;
			this.match(DaphneDSLGrammarParser.KW_IN);
			this.state = 145;
			_localctx._from = this.expr(0);
			this.state = 146;
			this.match(DaphneDSLGrammarParser.T__8);
			this.state = 147;
			_localctx._to = this.expr(0);
			this.state = 150;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.T__8) {
				{
				this.state = 148;
				this.match(DaphneDSLGrammarParser.T__8);
				this.state = 149;
				_localctx._step = this.expr(0);
				}
			}

			this.state = 152;
			this.match(DaphneDSLGrammarParser.T__7);
			this.state = 153;
			_localctx._bodyStmt = this.statement();
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public functionStatement(): FunctionStatementContext {
		let _localctx: FunctionStatementContext = new FunctionStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 18, DaphneDSLGrammarParser.RULE_functionStatement);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 155;
			this.match(DaphneDSLGrammarParser.KW_DEF);
			this.state = 156;
			_localctx._name = this.match(DaphneDSLGrammarParser.IDENTIFIER);
			this.state = 157;
			this.match(DaphneDSLGrammarParser.T__6);
			this.state = 159;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.IDENTIFIER) {
				{
				this.state = 158;
				_localctx._args = this.functionArgs();
				}
			}

			this.state = 161;
			this.match(DaphneDSLGrammarParser.T__7);
			this.state = 164;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.T__9) {
				{
				this.state = 162;
				this.match(DaphneDSLGrammarParser.T__9);
				this.state = 163;
				_localctx._retTys = this.functionRetTypes();
				}
			}

			this.state = 166;
			_localctx._bodyStmt = this.blockStatement();
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public returnStatement(): ReturnStatementContext {
		let _localctx: ReturnStatementContext = new ReturnStatementContext(this._ctx, this.state);
		this.enterRule(_localctx, 20, DaphneDSLGrammarParser.RULE_returnStatement);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 168;
			this.match(DaphneDSLGrammarParser.KW_RETURN);
			this.state = 177;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
				{
				this.state = 169;
				this.expr(0);
				this.state = 174;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				while (_la === DaphneDSLGrammarParser.T__4) {
					{
					{
					this.state = 170;
					this.match(DaphneDSLGrammarParser.T__4);
					this.state = 171;
					this.expr(0);
					}
					}
					this.state = 176;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
				}
				}
			}

			this.state = 179;
			this.match(DaphneDSLGrammarParser.T__0);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public functionArgs(): FunctionArgsContext {
		let _localctx: FunctionArgsContext = new FunctionArgsContext(this._ctx, this.state);
		this.enterRule(_localctx, 22, DaphneDSLGrammarParser.RULE_functionArgs);
		let _la: number;
		try {
			let _alt: number;
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 181;
			this.functionArg();
			this.state = 186;
			this._errHandler.sync(this);
			_alt = this.interpreter.adaptivePredict(this._input, 18, this._ctx);
			while (_alt !== 2 && _alt !== ATN.INVALID_ALT_NUMBER) {
				if (_alt === 1) {
					{
					{
					this.state = 182;
					this.match(DaphneDSLGrammarParser.T__4);
					this.state = 183;
					this.functionArg();
					}
					}
				}
				this.state = 188;
				this._errHandler.sync(this);
				_alt = this.interpreter.adaptivePredict(this._input, 18, this._ctx);
			}
			this.state = 190;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.T__4) {
				{
				this.state = 189;
				this.match(DaphneDSLGrammarParser.T__4);
				}
			}

			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public functionArg(): FunctionArgContext {
		let _localctx: FunctionArgContext = new FunctionArgContext(this._ctx, this.state);
		this.enterRule(_localctx, 24, DaphneDSLGrammarParser.RULE_functionArg);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 192;
			_localctx._var = this.match(DaphneDSLGrammarParser.IDENTIFIER);
			this.state = 195;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if (_la === DaphneDSLGrammarParser.T__8) {
				{
				this.state = 193;
				this.match(DaphneDSLGrammarParser.T__8);
				this.state = 194;
				_localctx._ty = this.funcTypeDef();
				}
			}

			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public functionRetTypes(): FunctionRetTypesContext {
		let _localctx: FunctionRetTypesContext = new FunctionRetTypesContext(this._ctx, this.state);
		this.enterRule(_localctx, 26, DaphneDSLGrammarParser.RULE_functionRetTypes);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 197;
			this.funcTypeDef();
			this.state = 202;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			while (_la === DaphneDSLGrammarParser.T__4) {
				{
				{
				this.state = 198;
				this.match(DaphneDSLGrammarParser.T__4);
				this.state = 199;
				this.funcTypeDef();
				}
				}
				this.state = 204;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
			}
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public funcTypeDef(): FuncTypeDefContext {
		let _localctx: FuncTypeDefContext = new FuncTypeDefContext(this._ctx, this.state);
		this.enterRule(_localctx, 28, DaphneDSLGrammarParser.RULE_funcTypeDef);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 212;
			this._errHandler.sync(this);
			switch (this._input.LA(1)) {
			case DaphneDSLGrammarParser.DATA_TYPE:
				{
				this.state = 205;
				_localctx._dataTy = this.match(DaphneDSLGrammarParser.DATA_TYPE);
				this.state = 209;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if (_la === DaphneDSLGrammarParser.T__10) {
					{
					this.state = 206;
					this.match(DaphneDSLGrammarParser.T__10);
					this.state = 207;
					_localctx._elTy = this.match(DaphneDSLGrammarParser.VALUE_TYPE);
					this.state = 208;
					this.match(DaphneDSLGrammarParser.T__11);
					}
				}

				}
				break;
			case DaphneDSLGrammarParser.VALUE_TYPE:
				{
				this.state = 211;
				_localctx._scalarTy = this.match(DaphneDSLGrammarParser.VALUE_TYPE);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}

	public expr(): ExprContext;
	public expr(_p: number): ExprContext;
	// @RuleVersion(0)
	public expr(_p?: number): ExprContext {
		if (_p === undefined) {
			_p = 0;
		}

		let _parentctx: ParserRuleContext = this._ctx;
		let _parentState: number = this.state;
		let _localctx: ExprContext = new ExprContext(this._ctx, _parentState);
		let _prevctx: ExprContext = _localctx;
		let _startState: number = 30;
		this.enterRecursionRule(_localctx, 30, DaphneDSLGrammarParser.RULE_expr, _p);
		let _la: number;
		try {
			let _alt: number;
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 323;
			this._errHandler.sync(this);
			switch ( this.interpreter.adaptivePredict(this._input, 38, this._ctx) ) {
			case 1:
				{
				_localctx = new LiteralExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;

				this.state = 215;
				this.literal();
				}
				break;

			case 2:
				{
				_localctx = new ArgExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 216;
				this.match(DaphneDSLGrammarParser.T__12);
				this.state = 217;
				(_localctx as ArgExprContext)._arg = this.match(DaphneDSLGrammarParser.IDENTIFIER);
				}
				break;

			case 3:
				{
				_localctx = new IdentifierExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				{
				this.state = 222;
				this._errHandler.sync(this);
				_alt = this.interpreter.adaptivePredict(this._input, 24, this._ctx);
				while (_alt !== 2 && _alt !== ATN.INVALID_ALT_NUMBER) {
					if (_alt === 1) {
						{
						{
						this.state = 218;
						this.match(DaphneDSLGrammarParser.IDENTIFIER);
						this.state = 219;
						this.match(DaphneDSLGrammarParser.T__3);
						}
						}
					}
					this.state = 224;
					this._errHandler.sync(this);
					_alt = this.interpreter.adaptivePredict(this._input, 24, this._ctx);
				}
				this.state = 225;
				this.match(DaphneDSLGrammarParser.IDENTIFIER);
				}
				}
				break;

			case 4:
				{
				_localctx = new ParanthesesExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 226;
				this.match(DaphneDSLGrammarParser.T__6);
				this.state = 227;
				this.expr(0);
				this.state = 228;
				this.match(DaphneDSLGrammarParser.T__7);
				}
				break;

			case 5:
				{
				_localctx = new CallExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 234;
				this._errHandler.sync(this);
				_alt = this.interpreter.adaptivePredict(this._input, 25, this._ctx);
				while (_alt !== 2 && _alt !== ATN.INVALID_ALT_NUMBER) {
					if (_alt === 1) {
						{
						{
						this.state = 230;
						(_localctx as CallExprContext)._ns = this.match(DaphneDSLGrammarParser.IDENTIFIER);
						this.state = 231;
						this.match(DaphneDSLGrammarParser.T__3);
						}
						}
					}
					this.state = 236;
					this._errHandler.sync(this);
					_alt = this.interpreter.adaptivePredict(this._input, 25, this._ctx);
				}
				this.state = 237;
				(_localctx as CallExprContext)._func = this.match(DaphneDSLGrammarParser.IDENTIFIER);
				this.state = 240;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if (_la === DaphneDSLGrammarParser.T__13) {
					{
					this.state = 238;
					this.match(DaphneDSLGrammarParser.T__13);
					this.state = 239;
					(_localctx as CallExprContext)._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
					}
				}

				this.state = 242;
				this.match(DaphneDSLGrammarParser.T__6);
				this.state = 251;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
					{
					this.state = 243;
					this.expr(0);
					this.state = 248;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
					while (_la === DaphneDSLGrammarParser.T__4) {
						{
						{
						this.state = 244;
						this.match(DaphneDSLGrammarParser.T__4);
						this.state = 245;
						this.expr(0);
						}
						}
						this.state = 250;
						this._errHandler.sync(this);
						_la = this._input.LA(1);
					}
					}
				}

				this.state = 253;
				this.match(DaphneDSLGrammarParser.T__7);
				}
				break;

			case 6:
				{
				_localctx = new CastExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 254;
				this.match(DaphneDSLGrammarParser.KW_AS);
				this.state = 264;
				this._errHandler.sync(this);
				switch ( this.interpreter.adaptivePredict(this._input, 29, this._ctx) ) {
				case 1:
					{
					{
					this.state = 255;
					this.match(DaphneDSLGrammarParser.T__3);
					this.state = 256;
					this.match(DaphneDSLGrammarParser.DATA_TYPE);
					}
					}
					break;

				case 2:
					{
					{
					this.state = 257;
					this.match(DaphneDSLGrammarParser.T__3);
					this.state = 258;
					this.match(DaphneDSLGrammarParser.VALUE_TYPE);
					}
					}
					break;

				case 3:
					{
					{
					this.state = 259;
					this.match(DaphneDSLGrammarParser.T__3);
					this.state = 260;
					this.match(DaphneDSLGrammarParser.DATA_TYPE);
					this.state = 261;
					this.match(DaphneDSLGrammarParser.T__10);
					this.state = 262;
					this.match(DaphneDSLGrammarParser.VALUE_TYPE);
					this.state = 263;
					this.match(DaphneDSLGrammarParser.T__11);
					}
					}
					break;
				}
				this.state = 266;
				this.match(DaphneDSLGrammarParser.T__6);
				this.state = 267;
				this.expr(0);
				this.state = 268;
				this.match(DaphneDSLGrammarParser.T__7);
				}
				break;

			case 7:
				{
				_localctx = new MinusExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 270;
				(_localctx as MinusExprContext)._op = this._input.LT(1);
				_la = this._input.LA(1);
				if (!(_la === DaphneDSLGrammarParser.T__16 || _la === DaphneDSLGrammarParser.T__17)) {
					(_localctx as MinusExprContext)._op = this._errHandler.recoverInline(this);
				} else {
					if (this._input.LA(1) === Token.EOF) {
						this.matchedEOF = true;
					}

					this._errHandler.reportMatch(this);
					this.consume();
				}
				this.state = 271;
				(_localctx as MinusExprContext)._arg = this.expr(13);
				}
				break;

			case 8:
				{
				_localctx = new MatrixLiteralExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 272;
				this.match(DaphneDSLGrammarParser.T__30);
				this.state = 281;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
					{
					this.state = 273;
					this.expr(0);
					this.state = 278;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
					while (_la === DaphneDSLGrammarParser.T__4) {
						{
						{
						this.state = 274;
						this.match(DaphneDSLGrammarParser.T__4);
						this.state = 275;
						this.expr(0);
						}
						}
						this.state = 280;
						this._errHandler.sync(this);
						_la = this._input.LA(1);
					}
					}
				}

				this.state = 283;
				this.match(DaphneDSLGrammarParser.T__31);
				this.state = 293;
				this._errHandler.sync(this);
				switch ( this.interpreter.adaptivePredict(this._input, 34, this._ctx) ) {
				case 1:
					{
					this.state = 284;
					this.match(DaphneDSLGrammarParser.T__6);
					this.state = 286;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
					if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
						{
						this.state = 285;
						(_localctx as MatrixLiteralExprContext)._rows = this.expr(0);
						}
					}

					this.state = 288;
					this.match(DaphneDSLGrammarParser.T__4);
					this.state = 290;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
					if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
						{
						this.state = 289;
						(_localctx as MatrixLiteralExprContext)._cols = this.expr(0);
						}
					}

					this.state = 292;
					this.match(DaphneDSLGrammarParser.T__7);
					}
					break;
				}
				}
				break;

			case 9:
				{
				_localctx = new ColMajorFrameLiteralExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 295;
				this.match(DaphneDSLGrammarParser.T__1);
				this.state = 309;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
					{
					this.state = 296;
					(_localctx as ColMajorFrameLiteralExprContext)._expr = this.expr(0);
					(_localctx as ColMajorFrameLiteralExprContext)._labels.push((_localctx as ColMajorFrameLiteralExprContext)._expr);
					this.state = 297;
					this.match(DaphneDSLGrammarParser.T__8);
					this.state = 298;
					(_localctx as ColMajorFrameLiteralExprContext)._expr = this.expr(0);
					(_localctx as ColMajorFrameLiteralExprContext)._cols.push((_localctx as ColMajorFrameLiteralExprContext)._expr);
					this.state = 306;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
					while (_la === DaphneDSLGrammarParser.T__4) {
						{
						{
						this.state = 299;
						this.match(DaphneDSLGrammarParser.T__4);
						this.state = 300;
						(_localctx as ColMajorFrameLiteralExprContext)._expr = this.expr(0);
						(_localctx as ColMajorFrameLiteralExprContext)._labels.push((_localctx as ColMajorFrameLiteralExprContext)._expr);
						this.state = 301;
						this.match(DaphneDSLGrammarParser.T__8);
						this.state = 302;
						(_localctx as ColMajorFrameLiteralExprContext)._expr = this.expr(0);
						(_localctx as ColMajorFrameLiteralExprContext)._cols.push((_localctx as ColMajorFrameLiteralExprContext)._expr);
						}
						}
						this.state = 308;
						this._errHandler.sync(this);
						_la = this._input.LA(1);
					}
					}
				}

				this.state = 311;
				this.match(DaphneDSLGrammarParser.T__2);
				}
				break;

			case 10:
				{
				_localctx = new RowMajorFrameLiteralExprContext(_localctx);
				this._ctx = _localctx;
				_prevctx = _localctx;
				this.state = 312;
				this.match(DaphneDSLGrammarParser.T__1);
				this.state = 313;
				(_localctx as RowMajorFrameLiteralExprContext)._labels = this.frameRow();
				this.state = 318;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				while (_la === DaphneDSLGrammarParser.T__4) {
					{
					{
					this.state = 314;
					this.match(DaphneDSLGrammarParser.T__4);
					this.state = 315;
					(_localctx as RowMajorFrameLiteralExprContext)._frameRow = this.frameRow();
					(_localctx as RowMajorFrameLiteralExprContext)._rows.push((_localctx as RowMajorFrameLiteralExprContext)._frameRow);
					}
					}
					this.state = 320;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
				}
				this.state = 321;
				this.match(DaphneDSLGrammarParser.T__2);
				}
				break;
			}
			this._ctx._stop = this._input.tryLT(-1);
			this.state = 377;
			this._errHandler.sync(this);
			_alt = this.interpreter.adaptivePredict(this._input, 44, this._ctx);
			while (_alt !== 2 && _alt !== ATN.INVALID_ALT_NUMBER) {
				if (_alt === 1) {
					if (this._parseListeners != null) {
						this.triggerExitRuleEvent();
					}
					_prevctx = _localctx;
					{
					this.state = 375;
					this._errHandler.sync(this);
					switch ( this.interpreter.adaptivePredict(this._input, 43, this._ctx) ) {
					case 1:
						{
						_localctx = new MatmulExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as MatmulExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 325;
						if (!(this.precpred(this._ctx, 12))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 12)");
						}
						this.state = 326;
						(_localctx as MatmulExprContext)._op = this.match(DaphneDSLGrammarParser.T__18);
						this.state = 327;
						(_localctx as MatmulExprContext)._rhs = this.expr(13);
						}
						break;

					case 2:
						{
						_localctx = new PowExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as PowExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 328;
						if (!(this.precpred(this._ctx, 11))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 11)");
						}
						this.state = 329;
						(_localctx as PowExprContext)._op = this.match(DaphneDSLGrammarParser.T__19);
						this.state = 330;
						(_localctx as PowExprContext)._rhs = this.expr(12);
						}
						break;

					case 3:
						{
						_localctx = new ModExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as ModExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 331;
						if (!(this.precpred(this._ctx, 10))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 10)");
						}
						this.state = 332;
						(_localctx as ModExprContext)._op = this.match(DaphneDSLGrammarParser.T__20);
						this.state = 333;
						(_localctx as ModExprContext)._rhs = this.expr(11);
						}
						break;

					case 4:
						{
						_localctx = new MulExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as MulExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 334;
						if (!(this.precpred(this._ctx, 9))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 9)");
						}
						this.state = 335;
						(_localctx as MulExprContext)._op = this._input.LT(1);
						_la = this._input.LA(1);
						if (!(_la === DaphneDSLGrammarParser.T__21 || _la === DaphneDSLGrammarParser.T__22)) {
							(_localctx as MulExprContext)._op = this._errHandler.recoverInline(this);
						} else {
							if (this._input.LA(1) === Token.EOF) {
								this.matchedEOF = true;
							}

							this._errHandler.reportMatch(this);
							this.consume();
						}
						this.state = 338;
						this._errHandler.sync(this);
						_la = this._input.LA(1);
						if (_la === DaphneDSLGrammarParser.T__13) {
							{
							this.state = 336;
							this.match(DaphneDSLGrammarParser.T__13);
							this.state = 337;
							(_localctx as MulExprContext)._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
							}
						}

						this.state = 340;
						(_localctx as MulExprContext)._rhs = this.expr(10);
						}
						break;

					case 5:
						{
						_localctx = new AddExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as AddExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 341;
						if (!(this.precpred(this._ctx, 8))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 8)");
						}
						this.state = 342;
						(_localctx as AddExprContext)._op = this._input.LT(1);
						_la = this._input.LA(1);
						if (!(_la === DaphneDSLGrammarParser.T__16 || _la === DaphneDSLGrammarParser.T__17)) {
							(_localctx as AddExprContext)._op = this._errHandler.recoverInline(this);
						} else {
							if (this._input.LA(1) === Token.EOF) {
								this.matchedEOF = true;
							}

							this._errHandler.reportMatch(this);
							this.consume();
						}
						this.state = 345;
						this._errHandler.sync(this);
						_la = this._input.LA(1);
						if (_la === DaphneDSLGrammarParser.T__13) {
							{
							this.state = 343;
							this.match(DaphneDSLGrammarParser.T__13);
							this.state = 344;
							(_localctx as AddExprContext)._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
							}
						}

						this.state = 347;
						(_localctx as AddExprContext)._rhs = this.expr(9);
						}
						break;

					case 6:
						{
						_localctx = new CmpExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as CmpExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 348;
						if (!(this.precpred(this._ctx, 7))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 7)");
						}
						this.state = 349;
						(_localctx as CmpExprContext)._op = this._input.LT(1);
						_la = this._input.LA(1);
						if (!((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__10) | (1 << DaphneDSLGrammarParser.T__11) | (1 << DaphneDSLGrammarParser.T__23) | (1 << DaphneDSLGrammarParser.T__24) | (1 << DaphneDSLGrammarParser.T__25) | (1 << DaphneDSLGrammarParser.T__26))) !== 0))) {
							(_localctx as CmpExprContext)._op = this._errHandler.recoverInline(this);
						} else {
							if (this._input.LA(1) === Token.EOF) {
								this.matchedEOF = true;
							}

							this._errHandler.reportMatch(this);
							this.consume();
						}
						this.state = 350;
						(_localctx as CmpExprContext)._rhs = this.expr(8);
						}
						break;

					case 7:
						{
						_localctx = new ConjExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as ConjExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 351;
						if (!(this.precpred(this._ctx, 6))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 6)");
						}
						this.state = 352;
						(_localctx as ConjExprContext)._op = this.match(DaphneDSLGrammarParser.T__27);
						this.state = 353;
						(_localctx as ConjExprContext)._rhs = this.expr(7);
						}
						break;

					case 8:
						{
						_localctx = new DisjExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as DisjExprContext)._lhs = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 354;
						if (!(this.precpred(this._ctx, 5))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 5)");
						}
						this.state = 355;
						(_localctx as DisjExprContext)._op = this.match(DaphneDSLGrammarParser.T__28);
						this.state = 356;
						(_localctx as DisjExprContext)._rhs = this.expr(6);
						}
						break;

					case 9:
						{
						_localctx = new CondExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as CondExprContext)._cond = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 357;
						if (!(this.precpred(this._ctx, 4))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 4)");
						}
						this.state = 358;
						this.match(DaphneDSLGrammarParser.T__29);
						this.state = 359;
						(_localctx as CondExprContext)._thenExpr = this.expr(0);
						this.state = 360;
						this.match(DaphneDSLGrammarParser.T__8);
						this.state = 361;
						(_localctx as CondExprContext)._elseExpr = this.expr(5);
						}
						break;

					case 10:
						{
						_localctx = new RightIdxFilterExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as RightIdxFilterExprContext)._obj = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 363;
						if (!(this.precpred(this._ctx, 15))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 15)");
						}
						this.state = 364;
						this.match(DaphneDSLGrammarParser.T__14);
						this.state = 366;
						this._errHandler.sync(this);
						_la = this._input.LA(1);
						if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
							{
							this.state = 365;
							(_localctx as RightIdxFilterExprContext)._rows = this.expr(0);
							}
						}

						this.state = 368;
						this.match(DaphneDSLGrammarParser.T__4);
						this.state = 370;
						this._errHandler.sync(this);
						_la = this._input.LA(1);
						if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
							{
							this.state = 369;
							(_localctx as RightIdxFilterExprContext)._cols = this.expr(0);
							}
						}

						this.state = 372;
						this.match(DaphneDSLGrammarParser.T__15);
						}
						break;

					case 11:
						{
						_localctx = new RightIdxExtractExprContext(new ExprContext(_parentctx, _parentState));
						(_localctx as RightIdxExtractExprContext)._obj = _prevctx;
						this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
						this.state = 373;
						if (!(this.precpred(this._ctx, 14))) {
							throw this.createFailedPredicateException("this.precpred(this._ctx, 14)");
						}
						this.state = 374;
						(_localctx as RightIdxExtractExprContext)._idx = this.indexing();
						}
						break;
					}
					}
				}
				this.state = 379;
				this._errHandler.sync(this);
				_alt = this.interpreter.adaptivePredict(this._input, 44, this._ctx);
			}
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public frameRow(): FrameRowContext {
		let _localctx: FrameRowContext = new FrameRowContext(this._ctx, this.state);
		this.enterRule(_localctx, 32, DaphneDSLGrammarParser.RULE_frameRow);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 380;
			this.match(DaphneDSLGrammarParser.T__30);
			this.state = 389;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
				{
				this.state = 381;
				this.expr(0);
				this.state = 386;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				while (_la === DaphneDSLGrammarParser.T__4) {
					{
					{
					this.state = 382;
					this.match(DaphneDSLGrammarParser.T__4);
					this.state = 383;
					this.expr(0);
					}
					}
					this.state = 388;
					this._errHandler.sync(this);
					_la = this._input.LA(1);
				}
				}
			}

			this.state = 391;
			this.match(DaphneDSLGrammarParser.T__31);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public indexing(): IndexingContext {
		let _localctx: IndexingContext = new IndexingContext(this._ctx, this.state);
		this.enterRule(_localctx, 34, DaphneDSLGrammarParser.RULE_indexing);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 393;
			this.match(DaphneDSLGrammarParser.T__30);
			this.state = 395;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__8) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
				{
				this.state = 394;
				_localctx._rows = this.range();
				}
			}

			this.state = 397;
			this.match(DaphneDSLGrammarParser.T__4);
			this.state = 399;
			this._errHandler.sync(this);
			_la = this._input.LA(1);
			if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__8) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
				{
				this.state = 398;
				_localctx._cols = this.range();
				}
			}

			this.state = 401;
			this.match(DaphneDSLGrammarParser.T__31);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public range(): RangeContext {
		let _localctx: RangeContext = new RangeContext(this._ctx, this.state);
		this.enterRule(_localctx, 36, DaphneDSLGrammarParser.RULE_range);
		let _la: number;
		try {
			this.state = 411;
			this._errHandler.sync(this);
			switch ( this.interpreter.adaptivePredict(this._input, 51, this._ctx) ) {
			case 1:
				this.enterOuterAlt(_localctx, 1);
				{
				this.state = 403;
				_localctx._pos = this.expr(0);
				}
				break;

			case 2:
				this.enterOuterAlt(_localctx, 2);
				{
				{
				this.state = 405;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
					{
					this.state = 404;
					_localctx._posLowerIncl = this.expr(0);
					}
				}

				this.state = 407;
				this.match(DaphneDSLGrammarParser.T__8);
				this.state = 409;
				this._errHandler.sync(this);
				_la = this._input.LA(1);
				if ((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__1) | (1 << DaphneDSLGrammarParser.T__6) | (1 << DaphneDSLGrammarParser.T__12) | (1 << DaphneDSLGrammarParser.T__16) | (1 << DaphneDSLGrammarParser.T__17) | (1 << DaphneDSLGrammarParser.T__30))) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & ((1 << (DaphneDSLGrammarParser.KW_TRUE - 39)) | (1 << (DaphneDSLGrammarParser.KW_FALSE - 39)) | (1 << (DaphneDSLGrammarParser.KW_AS - 39)) | (1 << (DaphneDSLGrammarParser.INT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.FLOAT_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.STRING_LITERAL - 39)) | (1 << (DaphneDSLGrammarParser.IDENTIFIER - 39)))) !== 0)) {
					{
					this.state = 408;
					_localctx._posUpperExcl = this.expr(0);
					}
				}

				}
				}
				break;
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public literal(): LiteralContext {
		let _localctx: LiteralContext = new LiteralContext(this._ctx, this.state);
		this.enterRule(_localctx, 38, DaphneDSLGrammarParser.RULE_literal);
		try {
			this.state = 417;
			this._errHandler.sync(this);
			switch (this._input.LA(1)) {
			case DaphneDSLGrammarParser.INT_LITERAL:
				this.enterOuterAlt(_localctx, 1);
				{
				this.state = 413;
				this.match(DaphneDSLGrammarParser.INT_LITERAL);
				}
				break;
			case DaphneDSLGrammarParser.FLOAT_LITERAL:
				this.enterOuterAlt(_localctx, 2);
				{
				this.state = 414;
				this.match(DaphneDSLGrammarParser.FLOAT_LITERAL);
				}
				break;
			case DaphneDSLGrammarParser.KW_TRUE:
			case DaphneDSLGrammarParser.KW_FALSE:
				this.enterOuterAlt(_localctx, 3);
				{
				this.state = 415;
				_localctx._bl = this.boolLiteral();
				}
				break;
			case DaphneDSLGrammarParser.STRING_LITERAL:
				this.enterOuterAlt(_localctx, 4);
				{
				this.state = 416;
				this.match(DaphneDSLGrammarParser.STRING_LITERAL);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}
	// @RuleVersion(0)
	public boolLiteral(): BoolLiteralContext {
		let _localctx: BoolLiteralContext = new BoolLiteralContext(this._ctx, this.state);
		this.enterRule(_localctx, 40, DaphneDSLGrammarParser.RULE_boolLiteral);
		let _la: number;
		try {
			this.enterOuterAlt(_localctx, 1);
			{
			this.state = 419;
			_la = this._input.LA(1);
			if (!(_la === DaphneDSLGrammarParser.KW_TRUE || _la === DaphneDSLGrammarParser.KW_FALSE)) {
			this._errHandler.recoverInline(this);
			} else {
				if (this._input.LA(1) === Token.EOF) {
					this.matchedEOF = true;
				}

				this._errHandler.reportMatch(this);
				this.consume();
			}
			}
		}
		catch (re) {
			if (re instanceof RecognitionException) {
				_localctx.exception = re;
				this._errHandler.reportError(this, re);
				this._errHandler.recover(this, re);
			} else {
				throw re;
			}
		}
		finally {
			this.exitRule();
		}
		return _localctx;
	}

	public sempred(_localctx: RuleContext, ruleIndex: number, predIndex: number): boolean {
		switch (ruleIndex) {
		case 15:
			return this.expr_sempred(_localctx as ExprContext, predIndex);
		}
		return true;
	}
	private expr_sempred(_localctx: ExprContext, predIndex: number): boolean {
		switch (predIndex) {
		case 0:
			return this.precpred(this._ctx, 12);

		case 1:
			return this.precpred(this._ctx, 11);

		case 2:
			return this.precpred(this._ctx, 10);

		case 3:
			return this.precpred(this._ctx, 9);

		case 4:
			return this.precpred(this._ctx, 8);

		case 5:
			return this.precpred(this._ctx, 7);

		case 6:
			return this.precpred(this._ctx, 6);

		case 7:
			return this.precpred(this._ctx, 5);

		case 8:
			return this.precpred(this._ctx, 4);

		case 9:
			return this.precpred(this._ctx, 15);

		case 10:
			return this.precpred(this._ctx, 14);
		}
		return true;
	}

	public static readonly _serializedATN: string =
		"\x03\uC91D\uCABA\u058D\uAFBA\u4F53\u0607\uEA8B\uC241\x038\u01A8\x04\x02" +
		"\t\x02\x04\x03\t\x03\x04\x04\t\x04\x04\x05\t\x05\x04\x06\t\x06\x04\x07" +
		"\t\x07\x04\b\t\b\x04\t\t\t\x04\n\t\n\x04\v\t\v\x04\f\t\f\x04\r\t\r\x04" +
		"\x0E\t\x0E\x04\x0F\t\x0F\x04\x10\t\x10\x04\x11\t\x11\x04\x12\t\x12\x04" +
		"\x13\t\x13\x04\x14\t\x14\x04\x15\t\x15\x04\x16\t\x16\x03\x02\x07\x02." +
		"\n\x02\f\x02\x0E\x021\v\x02\x03\x02\x03\x02\x03\x03\x03\x03\x03\x03\x03" +
		"\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x05\x03>\n\x03\x03\x04\x03" +
		"\x04\x03\x04\x03\x04\x05\x04D\n\x04\x03\x04\x03\x04\x03\x05\x03\x05\x07" +
		"\x05J\n\x05\f\x05\x0E\x05M\v\x05\x03\x05\x03\x05\x05\x05Q\n\x05\x03\x06" +
		"\x03\x06\x03\x06\x03\x07\x03\x07\x07\x07X\n\x07\f\x07\x0E\x07[\v\x07\x03" +
		"\x07\x03\x07\x05\x07_\n\x07\x03\x07\x03\x07\x03\x07\x07\x07d\n\x07\f\x07" +
		"\x0E\x07g\v\x07\x03\x07\x03\x07\x05\x07k\n\x07\x07\x07m\n\x07\f\x07\x0E" +
		"\x07p\v\x07\x03\x07\x03\x07\x03\x07\x03\x07\x03\b\x03\b\x03\b\x03\b\x03" +
		"\b\x03\b\x03\b\x05\b}\n\b\x03\t\x03\t\x03\t\x03\t\x03\t\x03\t\x03\t\x03" +
		"\t\x03\t\x03\t\x03\t\x03\t\x03\t\x05\t\x8C\n\t\x05\t\x8E\n\t\x03\n\x03" +
		"\n\x03\n\x03\n\x03\n\x03\n\x03\n\x03\n\x03\n\x05\n\x99\n\n\x03\n\x03\n" +
		"\x03\n\x03\v\x03\v\x03\v\x03\v\x05\v\xA2\n\v\x03\v\x03\v\x03\v\x05\v\xA7" +
		"\n\v\x03\v\x03\v\x03\f\x03\f\x03\f\x03\f\x07\f\xAF\n\f\f\f\x0E\f\xB2\v" +
		"\f\x05\f\xB4\n\f\x03\f\x03\f\x03\r\x03\r\x03\r\x07\r\xBB\n\r\f\r\x0E\r" +
		"\xBE\v\r\x03\r\x05\r\xC1\n\r\x03\x0E\x03\x0E\x03\x0E\x05\x0E\xC6\n\x0E" +
		"\x03\x0F\x03\x0F\x03\x0F\x07\x0F\xCB\n\x0F\f\x0F\x0E\x0F\xCE\v\x0F\x03" +
		"\x10\x03\x10\x03\x10\x03\x10\x05\x10\xD4\n\x10\x03\x10\x05\x10\xD7\n\x10" +
		"\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x07\x11\xDF\n\x11\f\x11" +
		"\x0E\x11\xE2\v\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03" +
		"\x11\x07\x11\xEB\n\x11\f\x11\x0E\x11\xEE\v\x11\x03\x11\x03\x11\x03\x11" +
		"\x05\x11\xF3\n\x11\x03\x11\x03\x11\x03\x11\x03\x11\x07\x11\xF9\n\x11\f" +
		"\x11\x0E\x11\xFC\v\x11\x05\x11\xFE\n\x11\x03\x11\x03\x11\x03\x11\x03\x11" +
		"\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x05\x11\u010B" +
		"\n\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11" +
		"\x03\x11\x03\x11\x07\x11\u0117\n\x11\f\x11\x0E\x11\u011A\v\x11\x05\x11" +
		"\u011C\n\x11\x03\x11\x03\x11\x03\x11\x05\x11\u0121\n\x11\x03\x11\x03\x11" +
		"\x05\x11\u0125\n\x11\x03\x11\x05\x11\u0128\n\x11\x03\x11\x03\x11\x03\x11" +
		"\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x07\x11\u0133\n\x11\f" +
		"\x11\x0E\x11\u0136\v\x11\x05\x11\u0138\n\x11\x03\x11\x03\x11\x03\x11\x03" +
		"\x11\x03\x11\x07\x11\u013F\n\x11\f\x11\x0E\x11\u0142\v\x11\x03\x11\x03" +
		"\x11\x05\x11\u0146\n\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11" +
		"\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x05\x11\u0155" +
		"\n\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x05\x11\u015C\n\x11\x03" +
		"\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03" +
		"\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03\x11\x03" +
		"\x11\x05\x11\u0171\n\x11\x03\x11\x03\x11\x05\x11\u0175\n\x11\x03\x11\x03" +
		"\x11\x03\x11\x07\x11\u017A\n\x11\f\x11\x0E\x11\u017D\v\x11\x03\x12\x03" +
		"\x12\x03\x12\x03\x12\x07\x12\u0183\n\x12\f\x12\x0E\x12\u0186\v\x12\x05" +
		"\x12\u0188\n\x12\x03\x12\x03\x12\x03\x13\x03\x13\x05\x13\u018E\n\x13\x03" +
		"\x13\x03\x13\x05\x13\u0192\n\x13\x03\x13\x03\x13\x03\x14\x03\x14\x05\x14" +
		"\u0198\n\x14\x03\x14\x03\x14\x05\x14\u019C\n\x14\x05\x14\u019E\n\x14\x03" +
		"\x15\x03\x15\x03\x15\x03\x15\x05\x15\u01A4\n\x15\x03\x16\x03\x16\x03\x16" +
		"\x02\x02\x03 \x17\x02\x02\x04\x02\x06\x02\b\x02\n\x02\f\x02\x0E\x02\x10" +
		"\x02\x12\x02\x14\x02\x16\x02\x18\x02\x1A\x02\x1C\x02\x1E\x02 \x02\"\x02" +
		"$\x02&\x02(\x02*\x02\x02\x06\x03\x02\x13\x14\x03\x02\x18\x19\x04\x02\r" +
		"\x0E\x1A\x1D\x03\x02)*\x02\u01E2\x02/\x03\x02\x02\x02\x04=\x03\x02\x02" +
		"\x02\x06?\x03\x02\x02\x02\bG\x03\x02\x02\x02\nR\x03\x02\x02\x02\fY\x03" +
		"\x02\x02\x02\x0Eu\x03\x02\x02\x02\x10\x8D\x03\x02\x02\x02\x12\x8F\x03" +
		"\x02\x02\x02\x14\x9D\x03\x02\x02\x02\x16\xAA\x03\x02\x02\x02\x18\xB7\x03" +
		"\x02\x02\x02\x1A\xC2\x03\x02\x02\x02\x1C\xC7\x03\x02\x02\x02\x1E\xD6\x03" +
		"\x02\x02\x02 \u0145\x03\x02\x02\x02\"\u017E\x03\x02\x02\x02$\u018B\x03" +
		"\x02\x02\x02&\u019D\x03\x02\x02\x02(\u01A3\x03\x02\x02\x02*\u01A5\x03" +
		"\x02\x02\x02,.\x05\x04\x03\x02-,\x03\x02\x02\x02.1\x03\x02\x02\x02/-\x03" +
		"\x02\x02\x02/0\x03\x02\x02\x0202\x03\x02\x02\x021/\x03\x02\x02\x0223\x07" +
		"\x02\x02\x033\x03\x03\x02\x02\x024>\x05\b\x05\x025>\x05\n\x06\x026>\x05" +
		"\f\x07\x027>\x05\x0E\b\x028>\x05\x10\t\x029>\x05\x12\n\x02:>\x05\x14\v" +
		"\x02;>\x05\x16\f\x02<>\x05\x06\x04\x02=4\x03\x02\x02\x02=5\x03\x02\x02" +
		"\x02=6\x03\x02\x02\x02=7\x03\x02\x02\x02=8\x03\x02\x02\x02=9\x03\x02\x02" +
		"\x02=:\x03\x02\x02\x02=;\x03\x02\x02\x02=<\x03\x02\x02\x02>\x05\x03\x02" +
		"\x02\x02?@\x07.\x02\x02@C\x073\x02\x02AB\x07+\x02\x02BD\x073\x02\x02C" +
		"A\x03\x02\x02\x02CD\x03\x02\x02\x02DE\x03\x02\x02\x02EF\x07\x03\x02\x02" +
		"F\x07\x03\x02\x02\x02GK\x07\x04\x02\x02HJ\x05\x04\x03\x02IH\x03\x02\x02" +
		"\x02JM\x03\x02\x02\x02KI\x03\x02\x02\x02KL\x03\x02\x02\x02LN\x03\x02\x02" +
		"\x02MK\x03\x02\x02\x02NP\x07\x05\x02\x02OQ\x07\x03\x02\x02PO\x03\x02\x02" +
		"\x02PQ\x03\x02\x02\x02Q\t\x03\x02\x02\x02RS\x05 \x11\x02ST\x07\x03\x02" +
		"\x02T\v\x03\x02\x02\x02UV\x074\x02\x02VX\x07\x06\x02\x02WU\x03\x02\x02" +
		"\x02X[\x03\x02\x02\x02YW\x03\x02\x02\x02YZ\x03\x02\x02\x02Z\\\x03\x02" +
		"\x02\x02[Y\x03\x02\x02\x02\\^\x074\x02\x02]_\x05$\x13\x02^]\x03\x02\x02" +
		"\x02^_\x03\x02\x02\x02_n\x03\x02\x02\x02`e\x07\x07\x02\x02ab\x074\x02" +
		"\x02bd\x07\x06\x02\x02ca\x03\x02\x02\x02dg\x03\x02\x02\x02ec\x03\x02\x02" +
		"\x02ef\x03\x02\x02\x02fh\x03\x02\x02\x02ge\x03\x02\x02\x02hj\x074\x02" +
		"\x02ik\x05$\x13\x02ji\x03\x02\x02\x02jk\x03\x02\x02\x02km\x03\x02\x02" +
		"\x02l`\x03\x02\x02\x02mp\x03\x02\x02\x02nl\x03\x02\x02\x02no\x03\x02\x02" +
		"\x02oq\x03\x02\x02\x02pn\x03\x02\x02\x02qr\x07\b\x02\x02rs\x05 \x11\x02" +
		"st\x07\x03\x02\x02t\r\x03\x02\x02\x02uv\x07#\x02\x02vw\x07\t\x02\x02w" +
		"x\x05 \x11\x02xy\x07\n\x02\x02y|\x05\x04\x03\x02z{\x07$\x02\x02{}\x05" +
		"\x04\x03\x02|z\x03\x02\x02\x02|}\x03\x02\x02\x02}\x0F\x03\x02\x02\x02" +
		"~\x7F\x07%\x02\x02\x7F\x80\x07\t\x02\x02\x80\x81\x05 \x11\x02\x81\x82" +
		"\x07\n\x02\x02\x82\x83\x05\x04\x03\x02\x83\x8E\x03\x02\x02\x02\x84\x85" +
		"\x07&\x02\x02\x85\x86\x05\x04\x03\x02\x86\x87\x07%\x02\x02\x87\x88\x07" +
		"\t\x02\x02\x88\x89\x05 \x11\x02\x89\x8B\x07\n\x02\x02\x8A\x8C\x07\x03" +
		"\x02\x02\x8B\x8A\x03\x02\x02\x02\x8B\x8C\x03\x02\x02\x02\x8C\x8E\x03\x02" +
		"\x02\x02\x8D~\x03\x02\x02\x02\x8D\x84\x03\x02\x02\x02\x8E\x11\x03\x02" +
		"\x02\x02\x8F\x90\x07\'\x02\x02\x90\x91\x07\t\x02\x02\x91\x92\x074\x02" +
		"\x02\x92\x93\x07(\x02\x02\x93\x94\x05 \x11\x02\x94\x95\x07\v\x02\x02\x95" +
		"\x98\x05 \x11\x02\x96\x97\x07\v\x02\x02\x97\x99\x05 \x11\x02\x98\x96\x03" +
		"\x02\x02\x02\x98\x99\x03\x02\x02\x02\x99\x9A\x03\x02\x02\x02\x9A\x9B\x07" +
		"\n\x02\x02\x9B\x9C\x05\x04\x03\x02\x9C\x13\x03\x02\x02\x02\x9D\x9E\x07" +
		",\x02\x02\x9E\x9F\x074\x02\x02\x9F\xA1\x07\t\x02\x02\xA0\xA2\x05\x18\r" +
		"\x02\xA1\xA0\x03\x02\x02\x02\xA1\xA2\x03\x02\x02\x02\xA2\xA3\x03\x02\x02" +
		"\x02\xA3\xA6\x07\n\x02\x02\xA4\xA5\x07\f\x02\x02\xA5\xA7\x05\x1C\x0F\x02" +
		"\xA6\xA4\x03\x02\x02\x02\xA6\xA7\x03\x02\x02\x02\xA7\xA8\x03\x02\x02\x02" +
		"\xA8\xA9\x05\b\x05\x02\xA9\x15\x03\x02\x02\x02\xAA\xB3\x07-\x02\x02\xAB" +
		"\xB0\x05 \x11\x02\xAC\xAD\x07\x07\x02\x02\xAD\xAF\x05 \x11\x02\xAE\xAC" +
		"\x03\x02\x02\x02\xAF\xB2\x03\x02\x02\x02\xB0\xAE\x03\x02\x02\x02\xB0\xB1" +
		"\x03\x02\x02\x02\xB1\xB4\x03\x02\x02\x02\xB2\xB0\x03\x02\x02\x02\xB3\xAB" +
		"\x03\x02\x02\x02\xB3\xB4\x03\x02\x02\x02\xB4\xB5\x03\x02\x02\x02\xB5\xB6" +
		"\x07\x03\x02\x02\xB6\x17\x03\x02\x02\x02\xB7\xBC\x05\x1A\x0E\x02\xB8\xB9" +
		"\x07\x07\x02\x02\xB9\xBB\x05\x1A\x0E\x02\xBA\xB8\x03\x02\x02\x02\xBB\xBE" +
		"\x03\x02\x02\x02\xBC\xBA\x03\x02\x02\x02\xBC\xBD\x03\x02\x02\x02\xBD\xC0" +
		"\x03\x02\x02\x02\xBE\xBC\x03\x02\x02\x02\xBF\xC1\x07\x07\x02\x02\xC0\xBF" +
		"\x03\x02\x02\x02\xC0\xC1\x03\x02\x02\x02\xC1\x19\x03\x02\x02\x02\xC2\xC5" +
		"\x074\x02\x02\xC3\xC4\x07\v\x02\x02\xC4\xC6\x05\x1E\x10\x02\xC5\xC3\x03" +
		"\x02\x02\x02\xC5\xC6\x03\x02\x02\x02\xC6\x1B\x03\x02\x02\x02\xC7\xCC\x05" +
		"\x1E\x10\x02\xC8\xC9\x07\x07\x02\x02\xC9\xCB\x05\x1E\x10\x02\xCA\xC8\x03" +
		"\x02\x02\x02\xCB\xCE\x03\x02\x02\x02\xCC\xCA\x03\x02\x02\x02\xCC\xCD\x03" +
		"\x02\x02\x02\xCD\x1D\x03\x02\x02\x02\xCE\xCC\x03\x02\x02\x02\xCF\xD3\x07" +
		"/\x02\x02\xD0\xD1\x07\r\x02\x02\xD1\xD2\x070\x02\x02\xD2\xD4\x07\x0E\x02" +
		"\x02\xD3\xD0\x03\x02\x02\x02\xD3\xD4\x03\x02\x02\x02\xD4\xD7\x03\x02\x02" +
		"\x02\xD5\xD7\x070\x02\x02\xD6\xCF\x03\x02\x02\x02\xD6\xD5\x03\x02\x02" +
		"\x02\xD7\x1F\x03\x02\x02\x02\xD8\xD9\b\x11\x01\x02\xD9\u0146\x05(\x15" +
		"\x02\xDA\xDB\x07\x0F\x02\x02\xDB\u0146\x074\x02\x02\xDC\xDD\x074\x02\x02" +
		"\xDD\xDF\x07\x06\x02\x02\xDE\xDC\x03\x02\x02\x02\xDF\xE2\x03\x02\x02\x02" +
		"\xE0\xDE\x03\x02\x02\x02\xE0\xE1\x03\x02\x02\x02\xE1\xE3\x03\x02\x02\x02" +
		"\xE2\xE0\x03\x02\x02\x02\xE3\u0146\x074\x02\x02\xE4\xE5\x07\t\x02\x02" +
		"\xE5\xE6\x05 \x11\x02\xE6\xE7\x07\n\x02\x02\xE7\u0146\x03\x02\x02\x02" +
		"\xE8\xE9\x074\x02\x02\xE9\xEB\x07\x06\x02\x02\xEA\xE8\x03\x02\x02\x02" +
		"\xEB\xEE\x03\x02\x02\x02\xEC\xEA\x03\x02\x02\x02\xEC\xED\x03\x02\x02\x02" +
		"\xED\xEF\x03\x02\x02\x02\xEE\xEC\x03\x02\x02\x02\xEF\xF2\x074\x02\x02" +
		"\xF0\xF1\x07\x10\x02\x02\xF1\xF3\x074\x02\x02\xF2\xF0\x03\x02\x02\x02" +
		"\xF2\xF3\x03\x02\x02\x02\xF3\xF4\x03\x02\x02\x02\xF4\xFD\x07\t\x02\x02" +
		"\xF5\xFA\x05 \x11\x02\xF6\xF7\x07\x07\x02\x02\xF7\xF9\x05 \x11\x02\xF8" +
		"\xF6\x03\x02\x02\x02\xF9\xFC\x03\x02\x02\x02\xFA\xF8\x03\x02\x02\x02\xFA" +
		"\xFB\x03\x02\x02\x02\xFB\xFE\x03\x02\x02\x02\xFC\xFA\x03\x02\x02\x02\xFD" +
		"\xF5\x03\x02\x02\x02\xFD\xFE\x03\x02\x02\x02\xFE\xFF\x03\x02\x02\x02\xFF" +
		"\u0146\x07\n\x02\x02\u0100\u010A\x07+\x02\x02\u0101\u0102\x07\x06\x02" +
		"\x02\u0102\u010B\x07/\x02\x02\u0103\u0104\x07\x06\x02\x02\u0104\u010B" +
		"\x070\x02\x02\u0105\u0106\x07\x06\x02\x02\u0106\u0107\x07/\x02\x02\u0107" +
		"\u0108\x07\r\x02\x02\u0108\u0109\x070\x02\x02\u0109\u010B\x07\x0E\x02" +
		"\x02\u010A\u0101\x03\x02\x02\x02\u010A\u0103\x03\x02\x02\x02\u010A\u0105" +
		"\x03\x02\x02\x02\u010B\u010C\x03\x02\x02\x02\u010C\u010D\x07\t\x02\x02" +
		"\u010D\u010E\x05 \x11\x02\u010E\u010F\x07\n\x02\x02\u010F\u0146\x03\x02" +
		"\x02\x02\u0110\u0111\t\x02\x02\x02\u0111\u0146\x05 \x11\x0F\u0112\u011B" +
		"\x07!\x02\x02\u0113\u0118\x05 \x11\x02\u0114\u0115\x07\x07\x02\x02\u0115" +
		"\u0117\x05 \x11\x02\u0116\u0114\x03\x02\x02\x02\u0117\u011A\x03\x02\x02" +
		"\x02\u0118\u0116\x03\x02\x02\x02\u0118\u0119\x03\x02\x02\x02\u0119\u011C" +
		"\x03\x02\x02\x02\u011A\u0118\x03\x02\x02\x02\u011B\u0113\x03\x02\x02\x02" +
		"\u011B\u011C\x03\x02\x02\x02\u011C\u011D\x03\x02\x02\x02\u011D\u0127\x07" +
		"\"\x02\x02\u011E\u0120\x07\t\x02\x02\u011F\u0121\x05 \x11\x02\u0120\u011F" +
		"\x03\x02\x02\x02\u0120\u0121\x03\x02\x02\x02\u0121\u0122\x03\x02\x02\x02" +
		"\u0122\u0124\x07\x07\x02\x02\u0123\u0125\x05 \x11\x02\u0124\u0123\x03" +
		"\x02\x02\x02\u0124\u0125\x03\x02\x02\x02\u0125\u0126\x03\x02\x02\x02\u0126" +
		"\u0128\x07\n\x02\x02\u0127\u011E\x03\x02\x02\x02\u0127\u0128\x03\x02\x02" +
		"\x02\u0128\u0146\x03\x02\x02\x02\u0129\u0137\x07\x04\x02\x02\u012A\u012B" +
		"\x05 \x11\x02\u012B\u012C\x07\v\x02\x02\u012C\u0134\x05 \x11\x02\u012D" +
		"\u012E\x07\x07\x02\x02\u012E\u012F\x05 \x11\x02\u012F\u0130\x07\v\x02" +
		"\x02\u0130\u0131\x05 \x11\x02\u0131\u0133\x03\x02\x02\x02\u0132\u012D" +
		"\x03\x02\x02\x02\u0133\u0136\x03\x02\x02\x02\u0134\u0132\x03\x02\x02\x02" +
		"\u0134\u0135\x03\x02\x02\x02\u0135\u0138\x03\x02\x02\x02\u0136\u0134\x03" +
		"\x02\x02\x02\u0137\u012A\x03\x02\x02\x02\u0137\u0138\x03\x02\x02\x02\u0138" +
		"\u0139\x03\x02\x02\x02\u0139\u0146\x07\x05\x02\x02\u013A\u013B\x07\x04" +
		"\x02\x02\u013B\u0140\x05\"\x12\x02\u013C\u013D\x07\x07\x02\x02\u013D\u013F" +
		"\x05\"\x12\x02\u013E\u013C\x03\x02\x02\x02\u013F\u0142\x03\x02\x02\x02" +
		"\u0140\u013E\x03\x02\x02\x02\u0140\u0141\x03\x02\x02\x02\u0141\u0143\x03" +
		"\x02\x02\x02\u0142\u0140\x03\x02\x02\x02\u0143\u0144\x07\x05\x02\x02\u0144" +
		"\u0146\x03\x02\x02\x02\u0145\xD8\x03\x02\x02\x02\u0145\xDA\x03\x02\x02" +
		"\x02\u0145\xE0\x03\x02\x02\x02\u0145\xE4\x03\x02\x02\x02\u0145\xEC\x03" +
		"\x02\x02\x02\u0145\u0100\x03\x02\x02\x02\u0145\u0110\x03\x02\x02\x02\u0145" +
		"\u0112\x03\x02\x02\x02\u0145\u0129\x03\x02\x02\x02\u0145\u013A\x03\x02" +
		"\x02\x02\u0146\u017B\x03\x02\x02\x02\u0147\u0148\f\x0E\x02\x02\u0148\u0149" +
		"\x07\x15\x02\x02\u0149\u017A\x05 \x11\x0F\u014A\u014B\f\r\x02\x02\u014B" +
		"\u014C\x07\x16\x02\x02\u014C\u017A\x05 \x11\x0E\u014D\u014E\f\f\x02\x02" +
		"\u014E\u014F\x07\x17\x02\x02\u014F\u017A\x05 \x11\r\u0150\u0151\f\v\x02" +
		"\x02\u0151\u0154\t\x03\x02\x02\u0152\u0153\x07\x10\x02\x02\u0153\u0155" +
		"\x074\x02\x02\u0154\u0152\x03\x02\x02\x02\u0154\u0155\x03\x02\x02\x02" +
		"\u0155\u0156\x03\x02\x02\x02\u0156\u017A\x05 \x11\f\u0157\u0158\f\n\x02" +
		"\x02\u0158\u015B\t\x02\x02\x02\u0159\u015A\x07\x10\x02\x02\u015A\u015C" +
		"\x074\x02\x02\u015B\u0159\x03\x02\x02\x02\u015B\u015C\x03\x02\x02\x02" +
		"\u015C\u015D\x03\x02\x02\x02\u015D\u017A\x05 \x11\v\u015E\u015F\f\t\x02" +
		"\x02\u015F\u0160\t\x04\x02\x02\u0160\u017A\x05 \x11\n\u0161\u0162\f\b" +
		"\x02\x02\u0162\u0163\x07\x1E\x02\x02\u0163\u017A\x05 \x11\t\u0164\u0165" +
		"\f\x07\x02\x02\u0165\u0166\x07\x1F\x02\x02\u0166\u017A\x05 \x11\b\u0167" +
		"\u0168\f\x06\x02\x02\u0168\u0169\x07 \x02\x02\u0169\u016A\x05 \x11\x02" +
		"\u016A\u016B\x07\v\x02\x02\u016B\u016C\x05 \x11\x07\u016C\u017A\x03\x02" +
		"\x02\x02\u016D\u016E\f\x11\x02\x02\u016E\u0170\x07\x11\x02\x02\u016F\u0171" +
		"\x05 \x11\x02\u0170\u016F\x03\x02\x02\x02\u0170\u0171\x03\x02\x02\x02" +
		"\u0171\u0172\x03\x02\x02\x02\u0172\u0174\x07\x07\x02\x02\u0173\u0175\x05" +
		" \x11\x02\u0174\u0173\x03\x02\x02\x02\u0174\u0175\x03\x02\x02\x02\u0175" +
		"\u0176\x03\x02\x02\x02\u0176\u017A\x07\x12\x02\x02\u0177\u0178\f\x10\x02" +
		"\x02\u0178\u017A\x05$\x13\x02\u0179\u0147\x03\x02\x02\x02\u0179\u014A" +
		"\x03\x02\x02\x02\u0179\u014D\x03\x02\x02\x02\u0179\u0150\x03\x02\x02\x02" +
		"\u0179\u0157\x03\x02\x02\x02\u0179\u015E\x03\x02\x02\x02\u0179\u0161\x03" +
		"\x02\x02\x02\u0179\u0164\x03\x02\x02\x02\u0179\u0167\x03\x02\x02\x02\u0179" +
		"\u016D\x03\x02\x02\x02\u0179\u0177\x03\x02\x02\x02\u017A\u017D\x03\x02" +
		"\x02\x02\u017B\u0179\x03\x02\x02\x02\u017B\u017C\x03\x02\x02\x02\u017C" +
		"!\x03\x02\x02\x02\u017D\u017B\x03\x02\x02\x02\u017E\u0187\x07!\x02\x02" +
		"\u017F\u0184\x05 \x11\x02\u0180\u0181\x07\x07\x02\x02\u0181\u0183\x05" +
		" \x11\x02\u0182\u0180\x03\x02\x02\x02\u0183\u0186\x03\x02\x02\x02\u0184" +
		"\u0182\x03\x02\x02\x02\u0184\u0185\x03\x02\x02\x02\u0185\u0188\x03\x02" +
		"\x02\x02\u0186\u0184\x03\x02\x02\x02\u0187\u017F\x03\x02\x02\x02\u0187" +
		"\u0188\x03\x02\x02\x02\u0188\u0189\x03\x02\x02\x02\u0189\u018A\x07\"\x02" +
		"\x02\u018A#\x03\x02\x02\x02\u018B\u018D\x07!\x02\x02\u018C\u018E\x05&" +
		"\x14\x02\u018D\u018C\x03\x02\x02\x02\u018D\u018E\x03\x02\x02\x02\u018E" +
		"\u018F\x03\x02\x02\x02\u018F\u0191\x07\x07\x02\x02\u0190\u0192\x05&\x14" +
		"\x02\u0191\u0190\x03\x02\x02\x02\u0191\u0192\x03\x02\x02\x02\u0192\u0193" +
		"\x03\x02\x02\x02\u0193\u0194\x07\"\x02\x02\u0194%\x03\x02\x02\x02\u0195" +
		"\u019E\x05 \x11\x02\u0196\u0198\x05 \x11\x02\u0197\u0196\x03\x02\x02\x02" +
		"\u0197\u0198\x03\x02\x02\x02\u0198\u0199\x03\x02\x02\x02\u0199\u019B\x07" +
		"\v\x02\x02\u019A\u019C\x05 \x11\x02\u019B\u019A\x03\x02\x02\x02\u019B" +
		"\u019C\x03\x02\x02\x02\u019C\u019E\x03\x02\x02\x02\u019D\u0195\x03\x02" +
		"\x02\x02\u019D\u0197\x03\x02\x02\x02\u019E\'\x03\x02\x02\x02\u019F\u01A4" +
		"\x071\x02\x02\u01A0\u01A4\x072\x02\x02\u01A1\u01A4\x05*\x16\x02\u01A2" +
		"\u01A4\x073\x02\x02\u01A3\u019F\x03\x02\x02\x02\u01A3\u01A0\x03\x02\x02" +
		"\x02\u01A3\u01A1\x03\x02\x02\x02\u01A3\u01A2\x03\x02\x02\x02\u01A4)\x03" +
		"\x02\x02\x02\u01A5\u01A6\t\x05\x02\x02\u01A6+\x03\x02\x02\x027/=CKPY^" +
		"ejn|\x8B\x8D\x98\xA1\xA6\xB0\xB3\xBC\xC0\xC5\xCC\xD3\xD6\xE0\xEC\xF2\xFA" +
		"\xFD\u010A\u0118\u011B\u0120\u0124\u0127\u0134\u0137\u0140\u0145\u0154" +
		"\u015B\u0170\u0174\u0179\u017B\u0184\u0187\u018D\u0191\u0197\u019B\u019D" +
		"\u01A3";
	public static __ATN: ATN;
	public static get _ATN(): ATN {
		if (!DaphneDSLGrammarParser.__ATN) {
			DaphneDSLGrammarParser.__ATN = new ATNDeserializer().deserialize(Utils.toCharArray(DaphneDSLGrammarParser._serializedATN));
		}

		return DaphneDSLGrammarParser.__ATN;
	}

}

export class ScriptContext extends ParserRuleContext {
	public EOF(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.EOF, 0); }
	public statement(): StatementContext[];
	public statement(i: number): StatementContext;
	public statement(i?: number): StatementContext | StatementContext[] {
		if (i === undefined) {
			return this.getRuleContexts(StatementContext);
		} else {
			return this.getRuleContext(i, StatementContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_script; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitScript) {
			return visitor.visitScript(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class StatementContext extends ParserRuleContext {
	public blockStatement(): BlockStatementContext | undefined {
		return this.tryGetRuleContext(0, BlockStatementContext);
	}
	public exprStatement(): ExprStatementContext | undefined {
		return this.tryGetRuleContext(0, ExprStatementContext);
	}
	public assignStatement(): AssignStatementContext | undefined {
		return this.tryGetRuleContext(0, AssignStatementContext);
	}
	public ifStatement(): IfStatementContext | undefined {
		return this.tryGetRuleContext(0, IfStatementContext);
	}
	public whileStatement(): WhileStatementContext | undefined {
		return this.tryGetRuleContext(0, WhileStatementContext);
	}
	public forStatement(): ForStatementContext | undefined {
		return this.tryGetRuleContext(0, ForStatementContext);
	}
	public functionStatement(): FunctionStatementContext | undefined {
		return this.tryGetRuleContext(0, FunctionStatementContext);
	}
	public returnStatement(): ReturnStatementContext | undefined {
		return this.tryGetRuleContext(0, ReturnStatementContext);
	}
	public importStatement(): ImportStatementContext | undefined {
		return this.tryGetRuleContext(0, ImportStatementContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_statement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitStatement) {
			return visitor.visitStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class ImportStatementContext extends ParserRuleContext {
	public _filePath!: Token;
	public _alias!: Token;
	public KW_IMPORT(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.KW_IMPORT, 0); }
	public STRING_LITERAL(): TerminalNode[];
	public STRING_LITERAL(i: number): TerminalNode;
	public STRING_LITERAL(i?: number): TerminalNode | TerminalNode[] {
		if (i === undefined) {
			return this.getTokens(DaphneDSLGrammarParser.STRING_LITERAL);
		} else {
			return this.getToken(DaphneDSLGrammarParser.STRING_LITERAL, i);
		}
	}
	public KW_AS(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.KW_AS, 0); }
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_importStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitImportStatement) {
			return visitor.visitImportStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class BlockStatementContext extends ParserRuleContext {
	public statement(): StatementContext[];
	public statement(i: number): StatementContext;
	public statement(i?: number): StatementContext | StatementContext[] {
		if (i === undefined) {
			return this.getRuleContexts(StatementContext);
		} else {
			return this.getRuleContext(i, StatementContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_blockStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitBlockStatement) {
			return visitor.visitBlockStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class ExprStatementContext extends ParserRuleContext {
	public expr(): ExprContext {
		return this.getRuleContext(0, ExprContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_exprStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitExprStatement) {
			return visitor.visitExprStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class AssignStatementContext extends ParserRuleContext {
	public IDENTIFIER(): TerminalNode[];
	public IDENTIFIER(i: number): TerminalNode;
	public IDENTIFIER(i?: number): TerminalNode | TerminalNode[] {
		if (i === undefined) {
			return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
		} else {
			return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
		}
	}
	public expr(): ExprContext {
		return this.getRuleContext(0, ExprContext);
	}
	public indexing(): IndexingContext[];
	public indexing(i: number): IndexingContext;
	public indexing(i?: number): IndexingContext | IndexingContext[] {
		if (i === undefined) {
			return this.getRuleContexts(IndexingContext);
		} else {
			return this.getRuleContext(i, IndexingContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_assignStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitAssignStatement) {
			return visitor.visitAssignStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class IfStatementContext extends ParserRuleContext {
	public _cond!: ExprContext;
	public _thenStmt!: StatementContext;
	public _elseStmt!: StatementContext;
	public KW_IF(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.KW_IF, 0); }
	public expr(): ExprContext {
		return this.getRuleContext(0, ExprContext);
	}
	public statement(): StatementContext[];
	public statement(i: number): StatementContext;
	public statement(i?: number): StatementContext | StatementContext[] {
		if (i === undefined) {
			return this.getRuleContexts(StatementContext);
		} else {
			return this.getRuleContext(i, StatementContext);
		}
	}
	public KW_ELSE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.KW_ELSE, 0); }
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_ifStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitIfStatement) {
			return visitor.visitIfStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class WhileStatementContext extends ParserRuleContext {
	public _cond!: ExprContext;
	public _bodyStmt!: StatementContext;
	public KW_WHILE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.KW_WHILE, 0); }
	public KW_DO(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.KW_DO, 0); }
	public expr(): ExprContext | undefined {
		return this.tryGetRuleContext(0, ExprContext);
	}
	public statement(): StatementContext | undefined {
		return this.tryGetRuleContext(0, StatementContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_whileStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitWhileStatement) {
			return visitor.visitWhileStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class ForStatementContext extends ParserRuleContext {
	public _var!: Token;
	public _from!: ExprContext;
	public _to!: ExprContext;
	public _step!: ExprContext;
	public _bodyStmt!: StatementContext;
	public KW_FOR(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.KW_FOR, 0); }
	public KW_IN(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.KW_IN, 0); }
	public IDENTIFIER(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	public statement(): StatementContext {
		return this.getRuleContext(0, StatementContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_forStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitForStatement) {
			return visitor.visitForStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class FunctionStatementContext extends ParserRuleContext {
	public _name!: Token;
	public _args!: FunctionArgsContext;
	public _retTys!: FunctionRetTypesContext;
	public _bodyStmt!: BlockStatementContext;
	public KW_DEF(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.KW_DEF, 0); }
	public IDENTIFIER(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
	public blockStatement(): BlockStatementContext {
		return this.getRuleContext(0, BlockStatementContext);
	}
	public functionArgs(): FunctionArgsContext | undefined {
		return this.tryGetRuleContext(0, FunctionArgsContext);
	}
	public functionRetTypes(): FunctionRetTypesContext | undefined {
		return this.tryGetRuleContext(0, FunctionRetTypesContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_functionStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitFunctionStatement) {
			return visitor.visitFunctionStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class ReturnStatementContext extends ParserRuleContext {
	public KW_RETURN(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.KW_RETURN, 0); }
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_returnStatement; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitReturnStatement) {
			return visitor.visitReturnStatement(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class FunctionArgsContext extends ParserRuleContext {
	public functionArg(): FunctionArgContext[];
	public functionArg(i: number): FunctionArgContext;
	public functionArg(i?: number): FunctionArgContext | FunctionArgContext[] {
		if (i === undefined) {
			return this.getRuleContexts(FunctionArgContext);
		} else {
			return this.getRuleContext(i, FunctionArgContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_functionArgs; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitFunctionArgs) {
			return visitor.visitFunctionArgs(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class FunctionArgContext extends ParserRuleContext {
	public _var!: Token;
	public _ty!: FuncTypeDefContext;
	public IDENTIFIER(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
	public funcTypeDef(): FuncTypeDefContext | undefined {
		return this.tryGetRuleContext(0, FuncTypeDefContext);
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_functionArg; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitFunctionArg) {
			return visitor.visitFunctionArg(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class FunctionRetTypesContext extends ParserRuleContext {
	public funcTypeDef(): FuncTypeDefContext[];
	public funcTypeDef(i: number): FuncTypeDefContext;
	public funcTypeDef(i?: number): FuncTypeDefContext | FuncTypeDefContext[] {
		if (i === undefined) {
			return this.getRuleContexts(FuncTypeDefContext);
		} else {
			return this.getRuleContext(i, FuncTypeDefContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_functionRetTypes; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitFunctionRetTypes) {
			return visitor.visitFunctionRetTypes(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class FuncTypeDefContext extends ParserRuleContext {
	public _dataTy!: Token;
	public _elTy!: Token;
	public _scalarTy!: Token;
	public DATA_TYPE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.DATA_TYPE, 0); }
	public VALUE_TYPE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.VALUE_TYPE, 0); }
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_funcTypeDef; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitFuncTypeDef) {
			return visitor.visitFuncTypeDef(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class ExprContext extends ParserRuleContext {
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_expr; }
	public copyFrom(ctx: ExprContext): void {
		super.copyFrom(ctx);
	}
}
export class LiteralExprContext extends ExprContext {
	public literal(): LiteralContext {
		return this.getRuleContext(0, LiteralContext);
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitLiteralExpr) {
			return visitor.visitLiteralExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class ArgExprContext extends ExprContext {
	public _arg!: Token;
	public IDENTIFIER(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitArgExpr) {
			return visitor.visitArgExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class IdentifierExprContext extends ExprContext {
	public IDENTIFIER(): TerminalNode[];
	public IDENTIFIER(i: number): TerminalNode;
	public IDENTIFIER(i?: number): TerminalNode | TerminalNode[] {
		if (i === undefined) {
			return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
		} else {
			return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitIdentifierExpr) {
			return visitor.visitIdentifierExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class ParanthesesExprContext extends ExprContext {
	public expr(): ExprContext {
		return this.getRuleContext(0, ExprContext);
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitParanthesesExpr) {
			return visitor.visitParanthesesExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class CallExprContext extends ExprContext {
	public _ns!: Token;
	public _func!: Token;
	public _kernel!: Token;
	public IDENTIFIER(): TerminalNode[];
	public IDENTIFIER(i: number): TerminalNode;
	public IDENTIFIER(i?: number): TerminalNode | TerminalNode[] {
		if (i === undefined) {
			return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
		} else {
			return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
		}
	}
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitCallExpr) {
			return visitor.visitCallExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class CastExprContext extends ExprContext {
	public KW_AS(): TerminalNode { return this.getToken(DaphneDSLGrammarParser.KW_AS, 0); }
	public expr(): ExprContext {
		return this.getRuleContext(0, ExprContext);
	}
	public DATA_TYPE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.DATA_TYPE, 0); }
	public VALUE_TYPE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.VALUE_TYPE, 0); }
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitCastExpr) {
			return visitor.visitCastExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class RightIdxFilterExprContext extends ExprContext {
	public _obj!: ExprContext;
	public _rows!: ExprContext;
	public _cols!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitRightIdxFilterExpr) {
			return visitor.visitRightIdxFilterExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class RightIdxExtractExprContext extends ExprContext {
	public _obj!: ExprContext;
	public _idx!: IndexingContext;
	public expr(): ExprContext {
		return this.getRuleContext(0, ExprContext);
	}
	public indexing(): IndexingContext {
		return this.getRuleContext(0, IndexingContext);
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitRightIdxExtractExpr) {
			return visitor.visitRightIdxExtractExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class MinusExprContext extends ExprContext {
	public _op!: Token;
	public _arg!: ExprContext;
	public expr(): ExprContext {
		return this.getRuleContext(0, ExprContext);
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitMinusExpr) {
			return visitor.visitMinusExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class MatmulExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitMatmulExpr) {
			return visitor.visitMatmulExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class PowExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitPowExpr) {
			return visitor.visitPowExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class ModExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitModExpr) {
			return visitor.visitModExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class MulExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _kernel!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	public IDENTIFIER(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitMulExpr) {
			return visitor.visitMulExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class AddExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _kernel!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	public IDENTIFIER(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitAddExpr) {
			return visitor.visitAddExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class CmpExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitCmpExpr) {
			return visitor.visitCmpExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class ConjExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitConjExpr) {
			return visitor.visitConjExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class DisjExprContext extends ExprContext {
	public _lhs!: ExprContext;
	public _op!: Token;
	public _rhs!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitDisjExpr) {
			return visitor.visitDisjExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class CondExprContext extends ExprContext {
	public _cond!: ExprContext;
	public _thenExpr!: ExprContext;
	public _elseExpr!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitCondExpr) {
			return visitor.visitCondExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class MatrixLiteralExprContext extends ExprContext {
	public _rows!: ExprContext;
	public _cols!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitMatrixLiteralExpr) {
			return visitor.visitMatrixLiteralExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class ColMajorFrameLiteralExprContext extends ExprContext {
	public _expr!: ExprContext;
	public _labels: ExprContext[] = [];
	public _cols: ExprContext[] = [];
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitColMajorFrameLiteralExpr) {
			return visitor.visitColMajorFrameLiteralExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}
export class RowMajorFrameLiteralExprContext extends ExprContext {
	public _labels!: FrameRowContext;
	public _frameRow!: FrameRowContext;
	public _rows: FrameRowContext[] = [];
	public frameRow(): FrameRowContext[];
	public frameRow(i: number): FrameRowContext;
	public frameRow(i?: number): FrameRowContext | FrameRowContext[] {
		if (i === undefined) {
			return this.getRuleContexts(FrameRowContext);
		} else {
			return this.getRuleContext(i, FrameRowContext);
		}
	}
	constructor(ctx: ExprContext) {
		super(ctx.parent, ctx.invokingState);
		this.copyFrom(ctx);
	}
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitRowMajorFrameLiteralExpr) {
			return visitor.visitRowMajorFrameLiteralExpr(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class FrameRowContext extends ParserRuleContext {
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_frameRow; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitFrameRow) {
			return visitor.visitFrameRow(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class IndexingContext extends ParserRuleContext {
	public _rows!: RangeContext;
	public _cols!: RangeContext;
	public range(): RangeContext[];
	public range(i: number): RangeContext;
	public range(i?: number): RangeContext | RangeContext[] {
		if (i === undefined) {
			return this.getRuleContexts(RangeContext);
		} else {
			return this.getRuleContext(i, RangeContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_indexing; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitIndexing) {
			return visitor.visitIndexing(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class RangeContext extends ParserRuleContext {
	public _pos!: ExprContext;
	public _posLowerIncl!: ExprContext;
	public _posUpperExcl!: ExprContext;
	public expr(): ExprContext[];
	public expr(i: number): ExprContext;
	public expr(i?: number): ExprContext | ExprContext[] {
		if (i === undefined) {
			return this.getRuleContexts(ExprContext);
		} else {
			return this.getRuleContext(i, ExprContext);
		}
	}
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_range; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitRange) {
			return visitor.visitRange(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class LiteralContext extends ParserRuleContext {
	public _bl!: BoolLiteralContext;
	public INT_LITERAL(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.INT_LITERAL, 0); }
	public FLOAT_LITERAL(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.FLOAT_LITERAL, 0); }
	public boolLiteral(): BoolLiteralContext | undefined {
		return this.tryGetRuleContext(0, BoolLiteralContext);
	}
	public STRING_LITERAL(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.STRING_LITERAL, 0); }
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_literal; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitLiteral) {
			return visitor.visitLiteral(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


export class BoolLiteralContext extends ParserRuleContext {
	public KW_TRUE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.KW_TRUE, 0); }
	public KW_FALSE(): TerminalNode | undefined { return this.tryGetToken(DaphneDSLGrammarParser.KW_FALSE, 0); }
	constructor(parent: ParserRuleContext | undefined, invokingState: number) {
		super(parent, invokingState);
	}
	// @Override
	public get ruleIndex(): number { return DaphneDSLGrammarParser.RULE_boolLiteral; }
	// @Override
	public accept<Result>(visitor: DaphneDSLGrammarVisitor<Result>): Result {
		if (visitor.visitBoolLiteral) {
			return visitor.visitBoolLiteral(this);
		} else {
			return visitor.visitChildren(this);
		}
	}
}


