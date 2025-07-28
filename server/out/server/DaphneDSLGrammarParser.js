"use strict";
// Generated from ./server/DaphneDSLGrammar.g4 by ANTLR 4.9.0-SNAPSHOT
Object.defineProperty(exports, "__esModule", { value: true });
exports.BoolLiteralContext = exports.LiteralContext = exports.RangeContext = exports.IndexingContext = exports.FrameRowContext = exports.RowMajorFrameLiteralExprContext = exports.ColMajorFrameLiteralExprContext = exports.MatrixLiteralExprContext = exports.CondExprContext = exports.DisjExprContext = exports.ConjExprContext = exports.CmpExprContext = exports.AddExprContext = exports.MulExprContext = exports.ModExprContext = exports.PowExprContext = exports.MatmulExprContext = exports.MinusExprContext = exports.RightIdxExtractExprContext = exports.RightIdxFilterExprContext = exports.CastExprContext = exports.CallExprContext = exports.ParanthesesExprContext = exports.IdentifierExprContext = exports.ArgExprContext = exports.LiteralExprContext = exports.ExprContext = exports.FuncTypeDefContext = exports.FunctionRetTypesContext = exports.FunctionArgContext = exports.FunctionArgsContext = exports.ReturnStatementContext = exports.FunctionStatementContext = exports.ForStatementContext = exports.WhileStatementContext = exports.IfStatementContext = exports.AssignStatementContext = exports.ExprStatementContext = exports.BlockStatementContext = exports.ImportStatementContext = exports.StatementContext = exports.ScriptContext = exports.DaphneDSLGrammarParser = void 0;
const ATN_1 = require("antlr4ts/atn/ATN");
const ATNDeserializer_1 = require("antlr4ts/atn/ATNDeserializer");
const FailedPredicateException_1 = require("antlr4ts/FailedPredicateException");
const NoViableAltException_1 = require("antlr4ts/NoViableAltException");
const Parser_1 = require("antlr4ts/Parser");
const ParserRuleContext_1 = require("antlr4ts/ParserRuleContext");
const ParserATNSimulator_1 = require("antlr4ts/atn/ParserATNSimulator");
const RecognitionException_1 = require("antlr4ts/RecognitionException");
const Token_1 = require("antlr4ts/Token");
const VocabularyImpl_1 = require("antlr4ts/VocabularyImpl");
const Utils = require("antlr4ts/misc/Utils");
class DaphneDSLGrammarParser extends Parser_1.Parser {
    // @Override
    // @NotNull
    get vocabulary() {
        return DaphneDSLGrammarParser.VOCABULARY;
    }
    // tslint:enable:no-trailing-whitespace
    // @Override
    get grammarFileName() { return "DaphneDSLGrammar.g4"; }
    // @Override
    get ruleNames() { return DaphneDSLGrammarParser.ruleNames; }
    // @Override
    get serializedATN() { return DaphneDSLGrammarParser._serializedATN; }
    createFailedPredicateException(predicate, message) {
        return new FailedPredicateException_1.FailedPredicateException(this, predicate, message);
    }
    constructor(input) {
        super(input);
        this._interp = new ParserATNSimulator_1.ParserATNSimulator(DaphneDSLGrammarParser._ATN, this);
    }
    // @RuleVersion(0)
    script() {
        let _localctx = new ScriptContext(this._ctx, this.state);
        this.enterRule(_localctx, 0, DaphneDSLGrammarParser.RULE_script);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    statement() {
        let _localctx = new StatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 2, DaphneDSLGrammarParser.RULE_statement);
        try {
            this.state = 59;
            this._errHandler.sync(this);
            switch (this.interpreter.adaptivePredict(this._input, 1, this._ctx)) {
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    importStatement() {
        let _localctx = new ImportStatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 4, DaphneDSLGrammarParser.RULE_importStatement);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    blockStatement() {
        let _localctx = new BlockStatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 6, DaphneDSLGrammarParser.RULE_blockStatement);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    exprStatement() {
        let _localctx = new ExprStatementContext(this._ctx, this.state);
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    assignStatement() {
        let _localctx = new AssignStatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 10, DaphneDSLGrammarParser.RULE_assignStatement);
        let _la;
        try {
            let _alt;
            this.enterOuterAlt(_localctx, 1);
            {
                this.state = 87;
                this._errHandler.sync(this);
                _alt = this.interpreter.adaptivePredict(this._input, 5, this._ctx);
                while (_alt !== 2 && _alt !== ATN_1.ATN.INVALID_ALT_NUMBER) {
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
                            while (_alt !== 2 && _alt !== ATN_1.ATN.INVALID_ALT_NUMBER) {
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    ifStatement() {
        let _localctx = new IfStatementContext(this._ctx, this.state);
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
                switch (this.interpreter.adaptivePredict(this._input, 10, this._ctx)) {
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    whileStatement() {
        let _localctx = new WhileStatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 14, DaphneDSLGrammarParser.RULE_whileStatement);
        let _la;
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
                        throw new NoViableAltException_1.NoViableAltException(this);
                }
            }
        }
        catch (re) {
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    forStatement() {
        let _localctx = new ForStatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 16, DaphneDSLGrammarParser.RULE_forStatement);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    functionStatement() {
        let _localctx = new FunctionStatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 18, DaphneDSLGrammarParser.RULE_functionStatement);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    returnStatement() {
        let _localctx = new ReturnStatementContext(this._ctx, this.state);
        this.enterRule(_localctx, 20, DaphneDSLGrammarParser.RULE_returnStatement);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    functionArgs() {
        let _localctx = new FunctionArgsContext(this._ctx, this.state);
        this.enterRule(_localctx, 22, DaphneDSLGrammarParser.RULE_functionArgs);
        let _la;
        try {
            let _alt;
            this.enterOuterAlt(_localctx, 1);
            {
                this.state = 181;
                this.functionArg();
                this.state = 186;
                this._errHandler.sync(this);
                _alt = this.interpreter.adaptivePredict(this._input, 18, this._ctx);
                while (_alt !== 2 && _alt !== ATN_1.ATN.INVALID_ALT_NUMBER) {
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    functionArg() {
        let _localctx = new FunctionArgContext(this._ctx, this.state);
        this.enterRule(_localctx, 24, DaphneDSLGrammarParser.RULE_functionArg);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    functionRetTypes() {
        let _localctx = new FunctionRetTypesContext(this._ctx, this.state);
        this.enterRule(_localctx, 26, DaphneDSLGrammarParser.RULE_functionRetTypes);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    funcTypeDef() {
        let _localctx = new FuncTypeDefContext(this._ctx, this.state);
        this.enterRule(_localctx, 28, DaphneDSLGrammarParser.RULE_funcTypeDef);
        let _la;
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
                        throw new NoViableAltException_1.NoViableAltException(this);
                }
            }
        }
        catch (re) {
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    expr(_p) {
        if (_p === undefined) {
            _p = 0;
        }
        let _parentctx = this._ctx;
        let _parentState = this.state;
        let _localctx = new ExprContext(this._ctx, _parentState);
        let _prevctx = _localctx;
        let _startState = 30;
        this.enterRecursionRule(_localctx, 30, DaphneDSLGrammarParser.RULE_expr, _p);
        let _la;
        try {
            let _alt;
            this.enterOuterAlt(_localctx, 1);
            {
                this.state = 323;
                this._errHandler.sync(this);
                switch (this.interpreter.adaptivePredict(this._input, 38, this._ctx)) {
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
                            _localctx._arg = this.match(DaphneDSLGrammarParser.IDENTIFIER);
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
                                while (_alt !== 2 && _alt !== ATN_1.ATN.INVALID_ALT_NUMBER) {
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
                            while (_alt !== 2 && _alt !== ATN_1.ATN.INVALID_ALT_NUMBER) {
                                if (_alt === 1) {
                                    {
                                        {
                                            this.state = 230;
                                            _localctx._ns = this.match(DaphneDSLGrammarParser.IDENTIFIER);
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
                            _localctx._func = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                            this.state = 240;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if (_la === DaphneDSLGrammarParser.T__13) {
                                {
                                    this.state = 238;
                                    this.match(DaphneDSLGrammarParser.T__13);
                                    this.state = 239;
                                    _localctx._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
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
                            switch (this.interpreter.adaptivePredict(this._input, 29, this._ctx)) {
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
                            _localctx._op = this._input.LT(1);
                            _la = this._input.LA(1);
                            if (!(_la === DaphneDSLGrammarParser.T__16 || _la === DaphneDSLGrammarParser.T__17)) {
                                _localctx._op = this._errHandler.recoverInline(this);
                            }
                            else {
                                if (this._input.LA(1) === Token_1.Token.EOF) {
                                    this.matchedEOF = true;
                                }
                                this._errHandler.reportMatch(this);
                                this.consume();
                            }
                            this.state = 271;
                            _localctx._arg = this.expr(13);
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
                            switch (this.interpreter.adaptivePredict(this._input, 34, this._ctx)) {
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
                                                _localctx._rows = this.expr(0);
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
                                                _localctx._cols = this.expr(0);
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
                                    _localctx._expr = this.expr(0);
                                    _localctx._labels.push(_localctx._expr);
                                    this.state = 297;
                                    this.match(DaphneDSLGrammarParser.T__8);
                                    this.state = 298;
                                    _localctx._expr = this.expr(0);
                                    _localctx._cols.push(_localctx._expr);
                                    this.state = 306;
                                    this._errHandler.sync(this);
                                    _la = this._input.LA(1);
                                    while (_la === DaphneDSLGrammarParser.T__4) {
                                        {
                                            {
                                                this.state = 299;
                                                this.match(DaphneDSLGrammarParser.T__4);
                                                this.state = 300;
                                                _localctx._expr = this.expr(0);
                                                _localctx._labels.push(_localctx._expr);
                                                this.state = 301;
                                                this.match(DaphneDSLGrammarParser.T__8);
                                                this.state = 302;
                                                _localctx._expr = this.expr(0);
                                                _localctx._cols.push(_localctx._expr);
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
                            _localctx._labels = this.frameRow();
                            this.state = 318;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            while (_la === DaphneDSLGrammarParser.T__4) {
                                {
                                    {
                                        this.state = 314;
                                        this.match(DaphneDSLGrammarParser.T__4);
                                        this.state = 315;
                                        _localctx._frameRow = this.frameRow();
                                        _localctx._rows.push(_localctx._frameRow);
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
                while (_alt !== 2 && _alt !== ATN_1.ATN.INVALID_ALT_NUMBER) {
                    if (_alt === 1) {
                        if (this._parseListeners != null) {
                            this.triggerExitRuleEvent();
                        }
                        _prevctx = _localctx;
                        {
                            this.state = 375;
                            this._errHandler.sync(this);
                            switch (this.interpreter.adaptivePredict(this._input, 43, this._ctx)) {
                                case 1:
                                    {
                                        _localctx = new MatmulExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 325;
                                        if (!(this.precpred(this._ctx, 12))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 12)");
                                        }
                                        this.state = 326;
                                        _localctx._op = this.match(DaphneDSLGrammarParser.T__18);
                                        this.state = 327;
                                        _localctx._rhs = this.expr(13);
                                    }
                                    break;
                                case 2:
                                    {
                                        _localctx = new PowExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 328;
                                        if (!(this.precpred(this._ctx, 11))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 11)");
                                        }
                                        this.state = 329;
                                        _localctx._op = this.match(DaphneDSLGrammarParser.T__19);
                                        this.state = 330;
                                        _localctx._rhs = this.expr(12);
                                    }
                                    break;
                                case 3:
                                    {
                                        _localctx = new ModExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 331;
                                        if (!(this.precpred(this._ctx, 10))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 10)");
                                        }
                                        this.state = 332;
                                        _localctx._op = this.match(DaphneDSLGrammarParser.T__20);
                                        this.state = 333;
                                        _localctx._rhs = this.expr(11);
                                    }
                                    break;
                                case 4:
                                    {
                                        _localctx = new MulExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 334;
                                        if (!(this.precpred(this._ctx, 9))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 9)");
                                        }
                                        this.state = 335;
                                        _localctx._op = this._input.LT(1);
                                        _la = this._input.LA(1);
                                        if (!(_la === DaphneDSLGrammarParser.T__21 || _la === DaphneDSLGrammarParser.T__22)) {
                                            _localctx._op = this._errHandler.recoverInline(this);
                                        }
                                        else {
                                            if (this._input.LA(1) === Token_1.Token.EOF) {
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
                                                _localctx._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                                            }
                                        }
                                        this.state = 340;
                                        _localctx._rhs = this.expr(10);
                                    }
                                    break;
                                case 5:
                                    {
                                        _localctx = new AddExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 341;
                                        if (!(this.precpred(this._ctx, 8))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 8)");
                                        }
                                        this.state = 342;
                                        _localctx._op = this._input.LT(1);
                                        _la = this._input.LA(1);
                                        if (!(_la === DaphneDSLGrammarParser.T__16 || _la === DaphneDSLGrammarParser.T__17)) {
                                            _localctx._op = this._errHandler.recoverInline(this);
                                        }
                                        else {
                                            if (this._input.LA(1) === Token_1.Token.EOF) {
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
                                                _localctx._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                                            }
                                        }
                                        this.state = 347;
                                        _localctx._rhs = this.expr(9);
                                    }
                                    break;
                                case 6:
                                    {
                                        _localctx = new CmpExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 348;
                                        if (!(this.precpred(this._ctx, 7))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 7)");
                                        }
                                        this.state = 349;
                                        _localctx._op = this._input.LT(1);
                                        _la = this._input.LA(1);
                                        if (!((((_la) & ~0x1F) === 0 && ((1 << _la) & ((1 << DaphneDSLGrammarParser.T__10) | (1 << DaphneDSLGrammarParser.T__11) | (1 << DaphneDSLGrammarParser.T__23) | (1 << DaphneDSLGrammarParser.T__24) | (1 << DaphneDSLGrammarParser.T__25) | (1 << DaphneDSLGrammarParser.T__26))) !== 0))) {
                                            _localctx._op = this._errHandler.recoverInline(this);
                                        }
                                        else {
                                            if (this._input.LA(1) === Token_1.Token.EOF) {
                                                this.matchedEOF = true;
                                            }
                                            this._errHandler.reportMatch(this);
                                            this.consume();
                                        }
                                        this.state = 350;
                                        _localctx._rhs = this.expr(8);
                                    }
                                    break;
                                case 7:
                                    {
                                        _localctx = new ConjExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 351;
                                        if (!(this.precpred(this._ctx, 6))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 6)");
                                        }
                                        this.state = 352;
                                        _localctx._op = this.match(DaphneDSLGrammarParser.T__27);
                                        this.state = 353;
                                        _localctx._rhs = this.expr(7);
                                    }
                                    break;
                                case 8:
                                    {
                                        _localctx = new DisjExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 354;
                                        if (!(this.precpred(this._ctx, 5))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 5)");
                                        }
                                        this.state = 355;
                                        _localctx._op = this.match(DaphneDSLGrammarParser.T__28);
                                        this.state = 356;
                                        _localctx._rhs = this.expr(6);
                                    }
                                    break;
                                case 9:
                                    {
                                        _localctx = new CondExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._cond = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 357;
                                        if (!(this.precpred(this._ctx, 4))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 4)");
                                        }
                                        this.state = 358;
                                        this.match(DaphneDSLGrammarParser.T__29);
                                        this.state = 359;
                                        _localctx._thenExpr = this.expr(0);
                                        this.state = 360;
                                        this.match(DaphneDSLGrammarParser.T__8);
                                        this.state = 361;
                                        _localctx._elseExpr = this.expr(5);
                                    }
                                    break;
                                case 10:
                                    {
                                        _localctx = new RightIdxFilterExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._obj = _prevctx;
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
                                                _localctx._rows = this.expr(0);
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
                                                _localctx._cols = this.expr(0);
                                            }
                                        }
                                        this.state = 372;
                                        this.match(DaphneDSLGrammarParser.T__15);
                                    }
                                    break;
                                case 11:
                                    {
                                        _localctx = new RightIdxExtractExprContext(new ExprContext(_parentctx, _parentState));
                                        _localctx._obj = _prevctx;
                                        this.pushNewRecursionContext(_localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 373;
                                        if (!(this.precpred(this._ctx, 14))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 14)");
                                        }
                                        this.state = 374;
                                        _localctx._idx = this.indexing();
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.unrollRecursionContexts(_parentctx);
        }
        return _localctx;
    }
    // @RuleVersion(0)
    frameRow() {
        let _localctx = new FrameRowContext(this._ctx, this.state);
        this.enterRule(_localctx, 32, DaphneDSLGrammarParser.RULE_frameRow);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    indexing() {
        let _localctx = new IndexingContext(this._ctx, this.state);
        this.enterRule(_localctx, 34, DaphneDSLGrammarParser.RULE_indexing);
        let _la;
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    range() {
        let _localctx = new RangeContext(this._ctx, this.state);
        this.enterRule(_localctx, 36, DaphneDSLGrammarParser.RULE_range);
        let _la;
        try {
            this.state = 411;
            this._errHandler.sync(this);
            switch (this.interpreter.adaptivePredict(this._input, 51, this._ctx)) {
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
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    literal() {
        let _localctx = new LiteralContext(this._ctx, this.state);
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
                    throw new NoViableAltException_1.NoViableAltException(this);
            }
        }
        catch (re) {
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    // @RuleVersion(0)
    boolLiteral() {
        let _localctx = new BoolLiteralContext(this._ctx, this.state);
        this.enterRule(_localctx, 40, DaphneDSLGrammarParser.RULE_boolLiteral);
        let _la;
        try {
            this.enterOuterAlt(_localctx, 1);
            {
                this.state = 419;
                _la = this._input.LA(1);
                if (!(_la === DaphneDSLGrammarParser.KW_TRUE || _la === DaphneDSLGrammarParser.KW_FALSE)) {
                    this._errHandler.recoverInline(this);
                }
                else {
                    if (this._input.LA(1) === Token_1.Token.EOF) {
                        this.matchedEOF = true;
                    }
                    this._errHandler.reportMatch(this);
                    this.consume();
                }
            }
        }
        catch (re) {
            if (re instanceof RecognitionException_1.RecognitionException) {
                _localctx.exception = re;
                this._errHandler.reportError(this, re);
                this._errHandler.recover(this, re);
            }
            else {
                throw re;
            }
        }
        finally {
            this.exitRule();
        }
        return _localctx;
    }
    sempred(_localctx, ruleIndex, predIndex) {
        switch (ruleIndex) {
            case 15:
                return this.expr_sempred(_localctx, predIndex);
        }
        return true;
    }
    expr_sempred(_localctx, predIndex) {
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
    static get _ATN() {
        if (!DaphneDSLGrammarParser.__ATN) {
            DaphneDSLGrammarParser.__ATN = new ATNDeserializer_1.ATNDeserializer().deserialize(Utils.toCharArray(DaphneDSLGrammarParser._serializedATN));
        }
        return DaphneDSLGrammarParser.__ATN;
    }
}
exports.DaphneDSLGrammarParser = DaphneDSLGrammarParser;
DaphneDSLGrammarParser.T__0 = 1;
DaphneDSLGrammarParser.T__1 = 2;
DaphneDSLGrammarParser.T__2 = 3;
DaphneDSLGrammarParser.T__3 = 4;
DaphneDSLGrammarParser.T__4 = 5;
DaphneDSLGrammarParser.T__5 = 6;
DaphneDSLGrammarParser.T__6 = 7;
DaphneDSLGrammarParser.T__7 = 8;
DaphneDSLGrammarParser.T__8 = 9;
DaphneDSLGrammarParser.T__9 = 10;
DaphneDSLGrammarParser.T__10 = 11;
DaphneDSLGrammarParser.T__11 = 12;
DaphneDSLGrammarParser.T__12 = 13;
DaphneDSLGrammarParser.T__13 = 14;
DaphneDSLGrammarParser.T__14 = 15;
DaphneDSLGrammarParser.T__15 = 16;
DaphneDSLGrammarParser.T__16 = 17;
DaphneDSLGrammarParser.T__17 = 18;
DaphneDSLGrammarParser.T__18 = 19;
DaphneDSLGrammarParser.T__19 = 20;
DaphneDSLGrammarParser.T__20 = 21;
DaphneDSLGrammarParser.T__21 = 22;
DaphneDSLGrammarParser.T__22 = 23;
DaphneDSLGrammarParser.T__23 = 24;
DaphneDSLGrammarParser.T__24 = 25;
DaphneDSLGrammarParser.T__25 = 26;
DaphneDSLGrammarParser.T__26 = 27;
DaphneDSLGrammarParser.T__27 = 28;
DaphneDSLGrammarParser.T__28 = 29;
DaphneDSLGrammarParser.T__29 = 30;
DaphneDSLGrammarParser.T__30 = 31;
DaphneDSLGrammarParser.T__31 = 32;
DaphneDSLGrammarParser.KW_IF = 33;
DaphneDSLGrammarParser.KW_ELSE = 34;
DaphneDSLGrammarParser.KW_WHILE = 35;
DaphneDSLGrammarParser.KW_DO = 36;
DaphneDSLGrammarParser.KW_FOR = 37;
DaphneDSLGrammarParser.KW_IN = 38;
DaphneDSLGrammarParser.KW_TRUE = 39;
DaphneDSLGrammarParser.KW_FALSE = 40;
DaphneDSLGrammarParser.KW_AS = 41;
DaphneDSLGrammarParser.KW_DEF = 42;
DaphneDSLGrammarParser.KW_RETURN = 43;
DaphneDSLGrammarParser.KW_IMPORT = 44;
DaphneDSLGrammarParser.DATA_TYPE = 45;
DaphneDSLGrammarParser.VALUE_TYPE = 46;
DaphneDSLGrammarParser.INT_LITERAL = 47;
DaphneDSLGrammarParser.FLOAT_LITERAL = 48;
DaphneDSLGrammarParser.STRING_LITERAL = 49;
DaphneDSLGrammarParser.IDENTIFIER = 50;
DaphneDSLGrammarParser.SCRIPT_STYLE_LINE_COMMENT = 51;
DaphneDSLGrammarParser.C_STYLE_LINE_COMMENT = 52;
DaphneDSLGrammarParser.MULTILINE_BLOCK_COMMENT = 53;
DaphneDSLGrammarParser.WS = 54;
DaphneDSLGrammarParser.RULE_script = 0;
DaphneDSLGrammarParser.RULE_statement = 1;
DaphneDSLGrammarParser.RULE_importStatement = 2;
DaphneDSLGrammarParser.RULE_blockStatement = 3;
DaphneDSLGrammarParser.RULE_exprStatement = 4;
DaphneDSLGrammarParser.RULE_assignStatement = 5;
DaphneDSLGrammarParser.RULE_ifStatement = 6;
DaphneDSLGrammarParser.RULE_whileStatement = 7;
DaphneDSLGrammarParser.RULE_forStatement = 8;
DaphneDSLGrammarParser.RULE_functionStatement = 9;
DaphneDSLGrammarParser.RULE_returnStatement = 10;
DaphneDSLGrammarParser.RULE_functionArgs = 11;
DaphneDSLGrammarParser.RULE_functionArg = 12;
DaphneDSLGrammarParser.RULE_functionRetTypes = 13;
DaphneDSLGrammarParser.RULE_funcTypeDef = 14;
DaphneDSLGrammarParser.RULE_expr = 15;
DaphneDSLGrammarParser.RULE_frameRow = 16;
DaphneDSLGrammarParser.RULE_indexing = 17;
DaphneDSLGrammarParser.RULE_range = 18;
DaphneDSLGrammarParser.RULE_literal = 19;
DaphneDSLGrammarParser.RULE_boolLiteral = 20;
// tslint:disable:no-trailing-whitespace
DaphneDSLGrammarParser.ruleNames = [
    "script", "statement", "importStatement", "blockStatement", "exprStatement",
    "assignStatement", "ifStatement", "whileStatement", "forStatement", "functionStatement",
    "returnStatement", "functionArgs", "functionArg", "functionRetTypes",
    "funcTypeDef", "expr", "frameRow", "indexing", "range", "literal", "boolLiteral",
];
DaphneDSLGrammarParser._LITERAL_NAMES = [
    undefined, "';'", "'{'", "'}'", "'.'", "','", "'='", "'('", "')'", "':'",
    "'->'", "'<'", "'>'", "'$'", "'::'", "'[['", "']]'", "'+'", "'-'", "'@'",
    "'^'", "'%'", "'*'", "'/'", "'=='", "'!='", "'<='", "'>='", "'&&'", "'||'",
    "'?'", "'['", "']'", "'if'", "'else'", "'while'", "'do'", "'for'", "'in'",
    "'true'", "'false'", "'as'", "'def'", "'return'", "'import'",
];
DaphneDSLGrammarParser._SYMBOLIC_NAMES = [
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
DaphneDSLGrammarParser.VOCABULARY = new VocabularyImpl_1.VocabularyImpl(DaphneDSLGrammarParser._LITERAL_NAMES, DaphneDSLGrammarParser._SYMBOLIC_NAMES, []);
DaphneDSLGrammarParser._serializedATN = "\x03\uC91D\uCABA\u058D\uAFBA\u4F53\u0607\uEA8B\uC241\x038\u01A8\x04\x02" +
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
class ScriptContext extends ParserRuleContext_1.ParserRuleContext {
    EOF() { return this.getToken(DaphneDSLGrammarParser.EOF, 0); }
    statement(i) {
        if (i === undefined) {
            return this.getRuleContexts(StatementContext);
        }
        else {
            return this.getRuleContext(i, StatementContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_script; }
    // @Override
    accept(visitor) {
        if (visitor.visitScript) {
            return visitor.visitScript(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ScriptContext = ScriptContext;
class StatementContext extends ParserRuleContext_1.ParserRuleContext {
    blockStatement() {
        return this.tryGetRuleContext(0, BlockStatementContext);
    }
    exprStatement() {
        return this.tryGetRuleContext(0, ExprStatementContext);
    }
    assignStatement() {
        return this.tryGetRuleContext(0, AssignStatementContext);
    }
    ifStatement() {
        return this.tryGetRuleContext(0, IfStatementContext);
    }
    whileStatement() {
        return this.tryGetRuleContext(0, WhileStatementContext);
    }
    forStatement() {
        return this.tryGetRuleContext(0, ForStatementContext);
    }
    functionStatement() {
        return this.tryGetRuleContext(0, FunctionStatementContext);
    }
    returnStatement() {
        return this.tryGetRuleContext(0, ReturnStatementContext);
    }
    importStatement() {
        return this.tryGetRuleContext(0, ImportStatementContext);
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_statement; }
    // @Override
    accept(visitor) {
        if (visitor.visitStatement) {
            return visitor.visitStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.StatementContext = StatementContext;
class ImportStatementContext extends ParserRuleContext_1.ParserRuleContext {
    KW_IMPORT() { return this.getToken(DaphneDSLGrammarParser.KW_IMPORT, 0); }
    STRING_LITERAL(i) {
        if (i === undefined) {
            return this.getTokens(DaphneDSLGrammarParser.STRING_LITERAL);
        }
        else {
            return this.getToken(DaphneDSLGrammarParser.STRING_LITERAL, i);
        }
    }
    KW_AS() { return this.tryGetToken(DaphneDSLGrammarParser.KW_AS, 0); }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_importStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitImportStatement) {
            return visitor.visitImportStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ImportStatementContext = ImportStatementContext;
class BlockStatementContext extends ParserRuleContext_1.ParserRuleContext {
    statement(i) {
        if (i === undefined) {
            return this.getRuleContexts(StatementContext);
        }
        else {
            return this.getRuleContext(i, StatementContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_blockStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitBlockStatement) {
            return visitor.visitBlockStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.BlockStatementContext = BlockStatementContext;
class ExprStatementContext extends ParserRuleContext_1.ParserRuleContext {
    expr() {
        return this.getRuleContext(0, ExprContext);
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_exprStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitExprStatement) {
            return visitor.visitExprStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ExprStatementContext = ExprStatementContext;
class AssignStatementContext extends ParserRuleContext_1.ParserRuleContext {
    IDENTIFIER(i) {
        if (i === undefined) {
            return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
        }
        else {
            return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
        }
    }
    expr() {
        return this.getRuleContext(0, ExprContext);
    }
    indexing(i) {
        if (i === undefined) {
            return this.getRuleContexts(IndexingContext);
        }
        else {
            return this.getRuleContext(i, IndexingContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_assignStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitAssignStatement) {
            return visitor.visitAssignStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.AssignStatementContext = AssignStatementContext;
class IfStatementContext extends ParserRuleContext_1.ParserRuleContext {
    KW_IF() { return this.getToken(DaphneDSLGrammarParser.KW_IF, 0); }
    expr() {
        return this.getRuleContext(0, ExprContext);
    }
    statement(i) {
        if (i === undefined) {
            return this.getRuleContexts(StatementContext);
        }
        else {
            return this.getRuleContext(i, StatementContext);
        }
    }
    KW_ELSE() { return this.tryGetToken(DaphneDSLGrammarParser.KW_ELSE, 0); }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_ifStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitIfStatement) {
            return visitor.visitIfStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.IfStatementContext = IfStatementContext;
class WhileStatementContext extends ParserRuleContext_1.ParserRuleContext {
    KW_WHILE() { return this.tryGetToken(DaphneDSLGrammarParser.KW_WHILE, 0); }
    KW_DO() { return this.tryGetToken(DaphneDSLGrammarParser.KW_DO, 0); }
    expr() {
        return this.tryGetRuleContext(0, ExprContext);
    }
    statement() {
        return this.tryGetRuleContext(0, StatementContext);
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_whileStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitWhileStatement) {
            return visitor.visitWhileStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.WhileStatementContext = WhileStatementContext;
class ForStatementContext extends ParserRuleContext_1.ParserRuleContext {
    KW_FOR() { return this.getToken(DaphneDSLGrammarParser.KW_FOR, 0); }
    KW_IN() { return this.getToken(DaphneDSLGrammarParser.KW_IN, 0); }
    IDENTIFIER() { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    statement() {
        return this.getRuleContext(0, StatementContext);
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_forStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitForStatement) {
            return visitor.visitForStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ForStatementContext = ForStatementContext;
class FunctionStatementContext extends ParserRuleContext_1.ParserRuleContext {
    KW_DEF() { return this.getToken(DaphneDSLGrammarParser.KW_DEF, 0); }
    IDENTIFIER() { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
    blockStatement() {
        return this.getRuleContext(0, BlockStatementContext);
    }
    functionArgs() {
        return this.tryGetRuleContext(0, FunctionArgsContext);
    }
    functionRetTypes() {
        return this.tryGetRuleContext(0, FunctionRetTypesContext);
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_functionStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitFunctionStatement) {
            return visitor.visitFunctionStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.FunctionStatementContext = FunctionStatementContext;
class ReturnStatementContext extends ParserRuleContext_1.ParserRuleContext {
    KW_RETURN() { return this.getToken(DaphneDSLGrammarParser.KW_RETURN, 0); }
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_returnStatement; }
    // @Override
    accept(visitor) {
        if (visitor.visitReturnStatement) {
            return visitor.visitReturnStatement(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ReturnStatementContext = ReturnStatementContext;
class FunctionArgsContext extends ParserRuleContext_1.ParserRuleContext {
    functionArg(i) {
        if (i === undefined) {
            return this.getRuleContexts(FunctionArgContext);
        }
        else {
            return this.getRuleContext(i, FunctionArgContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_functionArgs; }
    // @Override
    accept(visitor) {
        if (visitor.visitFunctionArgs) {
            return visitor.visitFunctionArgs(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.FunctionArgsContext = FunctionArgsContext;
class FunctionArgContext extends ParserRuleContext_1.ParserRuleContext {
    IDENTIFIER() { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
    funcTypeDef() {
        return this.tryGetRuleContext(0, FuncTypeDefContext);
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_functionArg; }
    // @Override
    accept(visitor) {
        if (visitor.visitFunctionArg) {
            return visitor.visitFunctionArg(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.FunctionArgContext = FunctionArgContext;
class FunctionRetTypesContext extends ParserRuleContext_1.ParserRuleContext {
    funcTypeDef(i) {
        if (i === undefined) {
            return this.getRuleContexts(FuncTypeDefContext);
        }
        else {
            return this.getRuleContext(i, FuncTypeDefContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_functionRetTypes; }
    // @Override
    accept(visitor) {
        if (visitor.visitFunctionRetTypes) {
            return visitor.visitFunctionRetTypes(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.FunctionRetTypesContext = FunctionRetTypesContext;
class FuncTypeDefContext extends ParserRuleContext_1.ParserRuleContext {
    DATA_TYPE() { return this.tryGetToken(DaphneDSLGrammarParser.DATA_TYPE, 0); }
    VALUE_TYPE() { return this.tryGetToken(DaphneDSLGrammarParser.VALUE_TYPE, 0); }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_funcTypeDef; }
    // @Override
    accept(visitor) {
        if (visitor.visitFuncTypeDef) {
            return visitor.visitFuncTypeDef(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.FuncTypeDefContext = FuncTypeDefContext;
class ExprContext extends ParserRuleContext_1.ParserRuleContext {
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_expr; }
    copyFrom(ctx) {
        super.copyFrom(ctx);
    }
}
exports.ExprContext = ExprContext;
class LiteralExprContext extends ExprContext {
    literal() {
        return this.getRuleContext(0, LiteralContext);
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitLiteralExpr) {
            return visitor.visitLiteralExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.LiteralExprContext = LiteralExprContext;
class ArgExprContext extends ExprContext {
    IDENTIFIER() { return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitArgExpr) {
            return visitor.visitArgExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ArgExprContext = ArgExprContext;
class IdentifierExprContext extends ExprContext {
    IDENTIFIER(i) {
        if (i === undefined) {
            return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
        }
        else {
            return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitIdentifierExpr) {
            return visitor.visitIdentifierExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.IdentifierExprContext = IdentifierExprContext;
class ParanthesesExprContext extends ExprContext {
    expr() {
        return this.getRuleContext(0, ExprContext);
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitParanthesesExpr) {
            return visitor.visitParanthesesExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ParanthesesExprContext = ParanthesesExprContext;
class CallExprContext extends ExprContext {
    IDENTIFIER(i) {
        if (i === undefined) {
            return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
        }
        else {
            return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
        }
    }
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitCallExpr) {
            return visitor.visitCallExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.CallExprContext = CallExprContext;
class CastExprContext extends ExprContext {
    KW_AS() { return this.getToken(DaphneDSLGrammarParser.KW_AS, 0); }
    expr() {
        return this.getRuleContext(0, ExprContext);
    }
    DATA_TYPE() { return this.tryGetToken(DaphneDSLGrammarParser.DATA_TYPE, 0); }
    VALUE_TYPE() { return this.tryGetToken(DaphneDSLGrammarParser.VALUE_TYPE, 0); }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitCastExpr) {
            return visitor.visitCastExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.CastExprContext = CastExprContext;
class RightIdxFilterExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitRightIdxFilterExpr) {
            return visitor.visitRightIdxFilterExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.RightIdxFilterExprContext = RightIdxFilterExprContext;
class RightIdxExtractExprContext extends ExprContext {
    expr() {
        return this.getRuleContext(0, ExprContext);
    }
    indexing() {
        return this.getRuleContext(0, IndexingContext);
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitRightIdxExtractExpr) {
            return visitor.visitRightIdxExtractExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.RightIdxExtractExprContext = RightIdxExtractExprContext;
class MinusExprContext extends ExprContext {
    expr() {
        return this.getRuleContext(0, ExprContext);
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitMinusExpr) {
            return visitor.visitMinusExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.MinusExprContext = MinusExprContext;
class MatmulExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitMatmulExpr) {
            return visitor.visitMatmulExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.MatmulExprContext = MatmulExprContext;
class PowExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitPowExpr) {
            return visitor.visitPowExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.PowExprContext = PowExprContext;
class ModExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitModExpr) {
            return visitor.visitModExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ModExprContext = ModExprContext;
class MulExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    IDENTIFIER() { return this.tryGetToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitMulExpr) {
            return visitor.visitMulExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.MulExprContext = MulExprContext;
class AddExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    IDENTIFIER() { return this.tryGetToken(DaphneDSLGrammarParser.IDENTIFIER, 0); }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitAddExpr) {
            return visitor.visitAddExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.AddExprContext = AddExprContext;
class CmpExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitCmpExpr) {
            return visitor.visitCmpExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.CmpExprContext = CmpExprContext;
class ConjExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitConjExpr) {
            return visitor.visitConjExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ConjExprContext = ConjExprContext;
class DisjExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitDisjExpr) {
            return visitor.visitDisjExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.DisjExprContext = DisjExprContext;
class CondExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitCondExpr) {
            return visitor.visitCondExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.CondExprContext = CondExprContext;
class MatrixLiteralExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitMatrixLiteralExpr) {
            return visitor.visitMatrixLiteralExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.MatrixLiteralExprContext = MatrixLiteralExprContext;
class ColMajorFrameLiteralExprContext extends ExprContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this._labels = [];
        this._cols = [];
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitColMajorFrameLiteralExpr) {
            return visitor.visitColMajorFrameLiteralExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.ColMajorFrameLiteralExprContext = ColMajorFrameLiteralExprContext;
class RowMajorFrameLiteralExprContext extends ExprContext {
    frameRow(i) {
        if (i === undefined) {
            return this.getRuleContexts(FrameRowContext);
        }
        else {
            return this.getRuleContext(i, FrameRowContext);
        }
    }
    constructor(ctx) {
        super(ctx.parent, ctx.invokingState);
        this._rows = [];
        this.copyFrom(ctx);
    }
    // @Override
    accept(visitor) {
        if (visitor.visitRowMajorFrameLiteralExpr) {
            return visitor.visitRowMajorFrameLiteralExpr(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.RowMajorFrameLiteralExprContext = RowMajorFrameLiteralExprContext;
class FrameRowContext extends ParserRuleContext_1.ParserRuleContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_frameRow; }
    // @Override
    accept(visitor) {
        if (visitor.visitFrameRow) {
            return visitor.visitFrameRow(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.FrameRowContext = FrameRowContext;
class IndexingContext extends ParserRuleContext_1.ParserRuleContext {
    range(i) {
        if (i === undefined) {
            return this.getRuleContexts(RangeContext);
        }
        else {
            return this.getRuleContext(i, RangeContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_indexing; }
    // @Override
    accept(visitor) {
        if (visitor.visitIndexing) {
            return visitor.visitIndexing(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.IndexingContext = IndexingContext;
class RangeContext extends ParserRuleContext_1.ParserRuleContext {
    expr(i) {
        if (i === undefined) {
            return this.getRuleContexts(ExprContext);
        }
        else {
            return this.getRuleContext(i, ExprContext);
        }
    }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_range; }
    // @Override
    accept(visitor) {
        if (visitor.visitRange) {
            return visitor.visitRange(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.RangeContext = RangeContext;
class LiteralContext extends ParserRuleContext_1.ParserRuleContext {
    INT_LITERAL() { return this.tryGetToken(DaphneDSLGrammarParser.INT_LITERAL, 0); }
    FLOAT_LITERAL() { return this.tryGetToken(DaphneDSLGrammarParser.FLOAT_LITERAL, 0); }
    boolLiteral() {
        return this.tryGetRuleContext(0, BoolLiteralContext);
    }
    STRING_LITERAL() { return this.tryGetToken(DaphneDSLGrammarParser.STRING_LITERAL, 0); }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_literal; }
    // @Override
    accept(visitor) {
        if (visitor.visitLiteral) {
            return visitor.visitLiteral(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.LiteralContext = LiteralContext;
class BoolLiteralContext extends ParserRuleContext_1.ParserRuleContext {
    KW_TRUE() { return this.tryGetToken(DaphneDSLGrammarParser.KW_TRUE, 0); }
    KW_FALSE() { return this.tryGetToken(DaphneDSLGrammarParser.KW_FALSE, 0); }
    constructor(parent, invokingState) {
        super(parent, invokingState);
    }
    // @Override
    get ruleIndex() { return DaphneDSLGrammarParser.RULE_boolLiteral; }
    // @Override
    accept(visitor) {
        if (visitor.visitBoolLiteral) {
            return visitor.visitBoolLiteral(this);
        }
        else {
            return visitor.visitChildren(this);
        }
    }
}
exports.BoolLiteralContext = BoolLiteralContext;
//# sourceMappingURL=DaphneDSLGrammarParser.js.map