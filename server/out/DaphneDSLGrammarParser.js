"use strict";
// Generated from DaphneDSLGrammar.g4 by ANTLR 4.13.2
// noinspection ES6UnusedImports,JSUnusedGlobalSymbols,JSUnusedLocalSymbols
Object.defineProperty(exports, "__esModule", { value: true });
exports.BoolLiteralContext = exports.LiteralContext = exports.RangeContext = exports.IndexingContext = exports.FrameRowContext = exports.IdentifierExprContext = exports.PowExprContext = exports.CallExprContext = exports.ColMajorFrameLiteralExprContext = exports.ArgExprContext = exports.MulExprContext = exports.RowMajorFrameLiteralExprContext = exports.LiteralExprContext = exports.AddExprContext = exports.CmpExprContext = exports.ParanthesesExprContext = exports.MinusExprContext = exports.MatrixLiteralExprContext = exports.RightIdxFilterExprContext = exports.DisjExprContext = exports.ConjExprContext = exports.CondExprContext = exports.MatmulExprContext = exports.CastExprContext = exports.ModExprContext = exports.RightIdxExtractExprContext = exports.ExprContext = exports.FuncTypeDefContext = exports.FunctionRetTypesContext = exports.FunctionArgContext = exports.FunctionArgsContext = exports.ReturnStatementContext = exports.FunctionStatementContext = exports.ForStatementContext = exports.WhileStatementContext = exports.IfStatementContext = exports.AssignStatementContext = exports.ExprStatementContext = exports.BlockStatementContext = exports.ImportStatementContext = exports.StatementContext = exports.ScriptContext = void 0;
const antlr4_1 = require("antlr4");
class DaphneDSLGrammarParser extends antlr4_1.Parser {
    get grammarFileName() { return "DaphneDSLGrammar.g4"; }
    get literalNames() { return DaphneDSLGrammarParser.literalNames; }
    get symbolicNames() { return DaphneDSLGrammarParser.symbolicNames; }
    get ruleNames() { return DaphneDSLGrammarParser.ruleNames; }
    get serializedATN() { return DaphneDSLGrammarParser._serializedATN; }
    createFailedPredicateException(predicate, message) {
        return new antlr4_1.FailedPredicateException(this, predicate, message);
    }
    constructor(input) {
        super(input);
        this._interp = new antlr4_1.ParserATNSimulator(this, DaphneDSLGrammarParser._ATN, DaphneDSLGrammarParser.DecisionsToDFA, new antlr4_1.PredictionContextCache());
    }
    // @RuleVersion(0)
    script() {
        const localctx = new ScriptContext(this, this._ctx, this.state);
        this.enterRule(localctx, 0, DaphneDSLGrammarParser.RULE_script);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 45;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                while ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 33)) & ~0x1F) === 0 && ((1 << (_la - 33)) & 249821) !== 0)) {
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
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    statement() {
        const localctx = new StatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 2, DaphneDSLGrammarParser.RULE_statement);
        try {
            this.state = 59;
            this._errHandler.sync(this);
            switch (this._interp.adaptivePredict(this._input, 1, this._ctx)) {
                case 1:
                    this.enterOuterAlt(localctx, 1);
                    {
                        this.state = 50;
                        this.blockStatement();
                    }
                    break;
                case 2:
                    this.enterOuterAlt(localctx, 2);
                    {
                        this.state = 51;
                        this.exprStatement();
                    }
                    break;
                case 3:
                    this.enterOuterAlt(localctx, 3);
                    {
                        this.state = 52;
                        this.assignStatement();
                    }
                    break;
                case 4:
                    this.enterOuterAlt(localctx, 4);
                    {
                        this.state = 53;
                        this.ifStatement();
                    }
                    break;
                case 5:
                    this.enterOuterAlt(localctx, 5);
                    {
                        this.state = 54;
                        this.whileStatement();
                    }
                    break;
                case 6:
                    this.enterOuterAlt(localctx, 6);
                    {
                        this.state = 55;
                        this.forStatement();
                    }
                    break;
                case 7:
                    this.enterOuterAlt(localctx, 7);
                    {
                        this.state = 56;
                        this.functionStatement();
                    }
                    break;
                case 8:
                    this.enterOuterAlt(localctx, 8);
                    {
                        this.state = 57;
                        this.returnStatement();
                    }
                    break;
                case 9:
                    this.enterOuterAlt(localctx, 9);
                    {
                        this.state = 58;
                        this.importStatement();
                    }
                    break;
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    importStatement() {
        const localctx = new ImportStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 4, DaphneDSLGrammarParser.RULE_importStatement);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 61;
                this.match(DaphneDSLGrammarParser.KW_IMPORT);
                this.state = 62;
                localctx._filePath = this.match(DaphneDSLGrammarParser.STRING_LITERAL);
                this.state = 65;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if (_la === 41) {
                    {
                        this.state = 63;
                        this.match(DaphneDSLGrammarParser.KW_AS);
                        this.state = 64;
                        localctx._alias = this.match(DaphneDSLGrammarParser.STRING_LITERAL);
                    }
                }
                this.state = 67;
                this.match(DaphneDSLGrammarParser.T__0);
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    blockStatement() {
        const localctx = new BlockStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 6, DaphneDSLGrammarParser.RULE_blockStatement);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 69;
                this.match(DaphneDSLGrammarParser.T__1);
                this.state = 73;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                while ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 33)) & ~0x1F) === 0 && ((1 << (_la - 33)) & 249821) !== 0)) {
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
                if (_la === 1) {
                    {
                        this.state = 77;
                        this.match(DaphneDSLGrammarParser.T__0);
                    }
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    exprStatement() {
        const localctx = new ExprStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 8, DaphneDSLGrammarParser.RULE_exprStatement);
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 80;
                this.expr(0);
                this.state = 81;
                this.match(DaphneDSLGrammarParser.T__0);
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    assignStatement() {
        const localctx = new AssignStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 10, DaphneDSLGrammarParser.RULE_assignStatement);
        let _la;
        try {
            let _alt;
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 87;
                this._errHandler.sync(this);
                _alt = this._interp.adaptivePredict(this._input, 5, this._ctx);
                while (_alt !== 2 && _alt !== antlr4_1.ATN.INVALID_ALT_NUMBER) {
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
                    _alt = this._interp.adaptivePredict(this._input, 5, this._ctx);
                }
                this.state = 90;
                this.match(DaphneDSLGrammarParser.IDENTIFIER);
                this.state = 92;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if (_la === 31) {
                    {
                        this.state = 91;
                        this.indexing();
                    }
                }
                this.state = 108;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                while (_la === 5) {
                    {
                        {
                            this.state = 94;
                            this.match(DaphneDSLGrammarParser.T__4);
                            this.state = 99;
                            this._errHandler.sync(this);
                            _alt = this._interp.adaptivePredict(this._input, 7, this._ctx);
                            while (_alt !== 2 && _alt !== antlr4_1.ATN.INVALID_ALT_NUMBER) {
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
                                _alt = this._interp.adaptivePredict(this._input, 7, this._ctx);
                            }
                            this.state = 102;
                            this.match(DaphneDSLGrammarParser.IDENTIFIER);
                            this.state = 104;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if (_la === 31) {
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
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    ifStatement() {
        const localctx = new IfStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 12, DaphneDSLGrammarParser.RULE_ifStatement);
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 115;
                this.match(DaphneDSLGrammarParser.KW_IF);
                this.state = 116;
                this.match(DaphneDSLGrammarParser.T__6);
                this.state = 117;
                localctx._cond = this.expr(0);
                this.state = 118;
                this.match(DaphneDSLGrammarParser.T__7);
                this.state = 119;
                localctx._thenStmt = this.statement();
                this.state = 122;
                this._errHandler.sync(this);
                switch (this._interp.adaptivePredict(this._input, 10, this._ctx)) {
                    case 1:
                        {
                            this.state = 120;
                            this.match(DaphneDSLGrammarParser.KW_ELSE);
                            this.state = 121;
                            localctx._elseStmt = this.statement();
                        }
                        break;
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    whileStatement() {
        const localctx = new WhileStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 14, DaphneDSLGrammarParser.RULE_whileStatement);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 139;
                this._errHandler.sync(this);
                switch (this._input.LA(1)) {
                    case 35:
                        {
                            this.state = 124;
                            this.match(DaphneDSLGrammarParser.KW_WHILE);
                            this.state = 125;
                            this.match(DaphneDSLGrammarParser.T__6);
                            this.state = 126;
                            localctx._cond = this.expr(0);
                            this.state = 127;
                            this.match(DaphneDSLGrammarParser.T__7);
                            this.state = 128;
                            localctx._bodyStmt = this.statement();
                        }
                        break;
                    case 36:
                        {
                            this.state = 130;
                            this.match(DaphneDSLGrammarParser.KW_DO);
                            this.state = 131;
                            localctx._bodyStmt = this.statement();
                            this.state = 132;
                            this.match(DaphneDSLGrammarParser.KW_WHILE);
                            this.state = 133;
                            this.match(DaphneDSLGrammarParser.T__6);
                            this.state = 134;
                            localctx._cond = this.expr(0);
                            this.state = 135;
                            this.match(DaphneDSLGrammarParser.T__7);
                            this.state = 137;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if (_la === 1) {
                                {
                                    this.state = 136;
                                    this.match(DaphneDSLGrammarParser.T__0);
                                }
                            }
                        }
                        break;
                    default:
                        throw new antlr4_1.NoViableAltException(this);
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    forStatement() {
        const localctx = new ForStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 16, DaphneDSLGrammarParser.RULE_forStatement);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 141;
                this.match(DaphneDSLGrammarParser.KW_FOR);
                this.state = 142;
                this.match(DaphneDSLGrammarParser.T__6);
                this.state = 143;
                localctx._var_ = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                this.state = 144;
                this.match(DaphneDSLGrammarParser.KW_IN);
                this.state = 145;
                localctx._from_ = this.expr(0);
                this.state = 146;
                this.match(DaphneDSLGrammarParser.T__8);
                this.state = 147;
                localctx._to = this.expr(0);
                this.state = 150;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if (_la === 9) {
                    {
                        this.state = 148;
                        this.match(DaphneDSLGrammarParser.T__8);
                        this.state = 149;
                        localctx._step = this.expr(0);
                    }
                }
                this.state = 152;
                this.match(DaphneDSLGrammarParser.T__7);
                this.state = 153;
                localctx._bodyStmt = this.statement();
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    functionStatement() {
        const localctx = new FunctionStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 18, DaphneDSLGrammarParser.RULE_functionStatement);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 155;
                this.match(DaphneDSLGrammarParser.KW_DEF);
                this.state = 156;
                localctx._name = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                this.state = 157;
                this.match(DaphneDSLGrammarParser.T__6);
                this.state = 159;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if (_la === 50) {
                    {
                        this.state = 158;
                        localctx._args = this.functionArgs();
                    }
                }
                this.state = 161;
                this.match(DaphneDSLGrammarParser.T__7);
                this.state = 164;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if (_la === 10) {
                    {
                        this.state = 162;
                        this.match(DaphneDSLGrammarParser.T__9);
                        this.state = 163;
                        localctx._retTys = this.functionRetTypes();
                    }
                }
                this.state = 166;
                localctx._bodyStmt = this.blockStatement();
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    returnStatement() {
        const localctx = new ReturnStatementContext(this, this._ctx, this.state);
        this.enterRule(localctx, 20, DaphneDSLGrammarParser.RULE_returnStatement);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 168;
                this.match(DaphneDSLGrammarParser.KW_RETURN);
                this.state = 177;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                    {
                        this.state = 169;
                        this.expr(0);
                        this.state = 174;
                        this._errHandler.sync(this);
                        _la = this._input.LA(1);
                        while (_la === 5) {
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
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    functionArgs() {
        const localctx = new FunctionArgsContext(this, this._ctx, this.state);
        this.enterRule(localctx, 22, DaphneDSLGrammarParser.RULE_functionArgs);
        let _la;
        try {
            let _alt;
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 181;
                this.functionArg();
                this.state = 186;
                this._errHandler.sync(this);
                _alt = this._interp.adaptivePredict(this._input, 18, this._ctx);
                while (_alt !== 2 && _alt !== antlr4_1.ATN.INVALID_ALT_NUMBER) {
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
                    _alt = this._interp.adaptivePredict(this._input, 18, this._ctx);
                }
                this.state = 190;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if (_la === 5) {
                    {
                        this.state = 189;
                        this.match(DaphneDSLGrammarParser.T__4);
                    }
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    functionArg() {
        const localctx = new FunctionArgContext(this, this._ctx, this.state);
        this.enterRule(localctx, 24, DaphneDSLGrammarParser.RULE_functionArg);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 192;
                localctx._var_ = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                this.state = 195;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if (_la === 9) {
                    {
                        this.state = 193;
                        this.match(DaphneDSLGrammarParser.T__8);
                        this.state = 194;
                        localctx._ty = this.funcTypeDef();
                    }
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    functionRetTypes() {
        const localctx = new FunctionRetTypesContext(this, this._ctx, this.state);
        this.enterRule(localctx, 26, DaphneDSLGrammarParser.RULE_functionRetTypes);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 197;
                this.funcTypeDef();
                this.state = 202;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                while (_la === 5) {
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
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    funcTypeDef() {
        const localctx = new FuncTypeDefContext(this, this._ctx, this.state);
        this.enterRule(localctx, 28, DaphneDSLGrammarParser.RULE_funcTypeDef);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 212;
                this._errHandler.sync(this);
                switch (this._input.LA(1)) {
                    case 45:
                        {
                            this.state = 205;
                            localctx._dataTy = this.match(DaphneDSLGrammarParser.DATA_TYPE);
                            this.state = 209;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if (_la === 11) {
                                {
                                    this.state = 206;
                                    this.match(DaphneDSLGrammarParser.T__10);
                                    this.state = 207;
                                    localctx._elTy = this.match(DaphneDSLGrammarParser.VALUE_TYPE);
                                    this.state = 208;
                                    this.match(DaphneDSLGrammarParser.T__11);
                                }
                            }
                        }
                        break;
                    case 46:
                        {
                            this.state = 211;
                            localctx._scalarTy = this.match(DaphneDSLGrammarParser.VALUE_TYPE);
                        }
                        break;
                    default:
                        throw new antlr4_1.NoViableAltException(this);
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    expr(_p) {
        if (_p === undefined) {
            _p = 0;
        }
        const _parentctx = this._ctx;
        const _parentState = this.state;
        let localctx = new ExprContext(this, this._ctx, _parentState);
        let _prevctx = localctx;
        const _startState = 30;
        this.enterRecursionRule(localctx, 30, DaphneDSLGrammarParser.RULE_expr, _p);
        let _la;
        try {
            let _alt;
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 323;
                this._errHandler.sync(this);
                switch (this._interp.adaptivePredict(this._input, 38, this._ctx)) {
                    case 1:
                        {
                            localctx = new LiteralExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 215;
                            this.literal();
                        }
                        break;
                    case 2:
                        {
                            localctx = new ArgExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 216;
                            this.match(DaphneDSLGrammarParser.T__12);
                            this.state = 217;
                            localctx._arg = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                        }
                        break;
                    case 3:
                        {
                            localctx = new IdentifierExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            {
                                this.state = 222;
                                this._errHandler.sync(this);
                                _alt = this._interp.adaptivePredict(this._input, 24, this._ctx);
                                while (_alt !== 2 && _alt !== antlr4_1.ATN.INVALID_ALT_NUMBER) {
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
                                    _alt = this._interp.adaptivePredict(this._input, 24, this._ctx);
                                }
                                this.state = 225;
                                this.match(DaphneDSLGrammarParser.IDENTIFIER);
                            }
                        }
                        break;
                    case 4:
                        {
                            localctx = new ParanthesesExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
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
                            localctx = new CallExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 234;
                            this._errHandler.sync(this);
                            _alt = this._interp.adaptivePredict(this._input, 25, this._ctx);
                            while (_alt !== 2 && _alt !== antlr4_1.ATN.INVALID_ALT_NUMBER) {
                                if (_alt === 1) {
                                    {
                                        {
                                            this.state = 230;
                                            localctx._ns = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                                            this.state = 231;
                                            this.match(DaphneDSLGrammarParser.T__3);
                                        }
                                    }
                                }
                                this.state = 236;
                                this._errHandler.sync(this);
                                _alt = this._interp.adaptivePredict(this._input, 25, this._ctx);
                            }
                            this.state = 237;
                            localctx._func = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                            this.state = 240;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if (_la === 14) {
                                {
                                    this.state = 238;
                                    this.match(DaphneDSLGrammarParser.T__13);
                                    this.state = 239;
                                    localctx._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                                }
                            }
                            this.state = 242;
                            this.match(DaphneDSLGrammarParser.T__6);
                            this.state = 251;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                {
                                    this.state = 243;
                                    this.expr(0);
                                    this.state = 248;
                                    this._errHandler.sync(this);
                                    _la = this._input.LA(1);
                                    while (_la === 5) {
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
                            localctx = new CastExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 254;
                            this.match(DaphneDSLGrammarParser.KW_AS);
                            this.state = 264;
                            this._errHandler.sync(this);
                            switch (this._interp.adaptivePredict(this._input, 29, this._ctx)) {
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
                            localctx = new MinusExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 270;
                            localctx._op = this._input.LT(1);
                            _la = this._input.LA(1);
                            if (!(_la === 17 || _la === 18)) {
                                localctx._op = this._errHandler.recoverInline(this);
                            }
                            else {
                                this._errHandler.reportMatch(this);
                                this.consume();
                            }
                            this.state = 271;
                            localctx._arg = this.expr(13);
                        }
                        break;
                    case 8:
                        {
                            localctx = new MatrixLiteralExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 272;
                            this.match(DaphneDSLGrammarParser.T__30);
                            this.state = 281;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                {
                                    this.state = 273;
                                    this.expr(0);
                                    this.state = 278;
                                    this._errHandler.sync(this);
                                    _la = this._input.LA(1);
                                    while (_la === 5) {
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
                            switch (this._interp.adaptivePredict(this._input, 34, this._ctx)) {
                                case 1:
                                    {
                                        this.state = 284;
                                        this.match(DaphneDSLGrammarParser.T__6);
                                        this.state = 286;
                                        this._errHandler.sync(this);
                                        _la = this._input.LA(1);
                                        if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                            {
                                                this.state = 285;
                                                localctx._rows = this.expr(0);
                                            }
                                        }
                                        this.state = 288;
                                        this.match(DaphneDSLGrammarParser.T__4);
                                        this.state = 290;
                                        this._errHandler.sync(this);
                                        _la = this._input.LA(1);
                                        if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                            {
                                                this.state = 289;
                                                localctx._cols = this.expr(0);
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
                            localctx = new ColMajorFrameLiteralExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 295;
                            this.match(DaphneDSLGrammarParser.T__1);
                            this.state = 309;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                {
                                    this.state = 296;
                                    localctx._expr = this.expr(0);
                                    localctx._labels.push(localctx._expr);
                                    this.state = 297;
                                    this.match(DaphneDSLGrammarParser.T__8);
                                    this.state = 298;
                                    localctx._expr = this.expr(0);
                                    localctx._cols.push(localctx._expr);
                                    this.state = 306;
                                    this._errHandler.sync(this);
                                    _la = this._input.LA(1);
                                    while (_la === 5) {
                                        {
                                            {
                                                this.state = 299;
                                                this.match(DaphneDSLGrammarParser.T__4);
                                                this.state = 300;
                                                localctx._expr = this.expr(0);
                                                localctx._labels.push(localctx._expr);
                                                this.state = 301;
                                                this.match(DaphneDSLGrammarParser.T__8);
                                                this.state = 302;
                                                localctx._expr = this.expr(0);
                                                localctx._cols.push(localctx._expr);
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
                            localctx = new RowMajorFrameLiteralExprContext(this, localctx);
                            this._ctx = localctx;
                            _prevctx = localctx;
                            this.state = 312;
                            this.match(DaphneDSLGrammarParser.T__1);
                            this.state = 313;
                            localctx._labels = this.frameRow();
                            this.state = 318;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            while (_la === 5) {
                                {
                                    {
                                        this.state = 314;
                                        this.match(DaphneDSLGrammarParser.T__4);
                                        this.state = 315;
                                        localctx._frameRow = this.frameRow();
                                        localctx._rows.push(localctx._frameRow);
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
                this._ctx.stop = this._input.LT(-1);
                this.state = 377;
                this._errHandler.sync(this);
                _alt = this._interp.adaptivePredict(this._input, 44, this._ctx);
                while (_alt !== 2 && _alt !== antlr4_1.ATN.INVALID_ALT_NUMBER) {
                    if (_alt === 1) {
                        if (this._parseListeners != null) {
                            this.triggerExitRuleEvent();
                        }
                        _prevctx = localctx;
                        {
                            this.state = 375;
                            this._errHandler.sync(this);
                            switch (this._interp.adaptivePredict(this._input, 43, this._ctx)) {
                                case 1:
                                    {
                                        localctx = new MatmulExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 325;
                                        if (!(this.precpred(this._ctx, 12))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 12)");
                                        }
                                        this.state = 326;
                                        localctx._op = this.match(DaphneDSLGrammarParser.T__18);
                                        this.state = 327;
                                        localctx._rhs = this.expr(13);
                                    }
                                    break;
                                case 2:
                                    {
                                        localctx = new PowExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 328;
                                        if (!(this.precpred(this._ctx, 11))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 11)");
                                        }
                                        this.state = 329;
                                        localctx._op = this.match(DaphneDSLGrammarParser.T__19);
                                        this.state = 330;
                                        localctx._rhs = this.expr(12);
                                    }
                                    break;
                                case 3:
                                    {
                                        localctx = new ModExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 331;
                                        if (!(this.precpred(this._ctx, 10))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 10)");
                                        }
                                        this.state = 332;
                                        localctx._op = this.match(DaphneDSLGrammarParser.T__20);
                                        this.state = 333;
                                        localctx._rhs = this.expr(11);
                                    }
                                    break;
                                case 4:
                                    {
                                        localctx = new MulExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 334;
                                        if (!(this.precpred(this._ctx, 9))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 9)");
                                        }
                                        this.state = 335;
                                        localctx._op = this._input.LT(1);
                                        _la = this._input.LA(1);
                                        if (!(_la === 22 || _la === 23)) {
                                            localctx._op = this._errHandler.recoverInline(this);
                                        }
                                        else {
                                            this._errHandler.reportMatch(this);
                                            this.consume();
                                        }
                                        this.state = 338;
                                        this._errHandler.sync(this);
                                        _la = this._input.LA(1);
                                        if (_la === 14) {
                                            {
                                                this.state = 336;
                                                this.match(DaphneDSLGrammarParser.T__13);
                                                this.state = 337;
                                                localctx._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                                            }
                                        }
                                        this.state = 340;
                                        localctx._rhs = this.expr(10);
                                    }
                                    break;
                                case 5:
                                    {
                                        localctx = new AddExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 341;
                                        if (!(this.precpred(this._ctx, 8))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 8)");
                                        }
                                        this.state = 342;
                                        localctx._op = this._input.LT(1);
                                        _la = this._input.LA(1);
                                        if (!(_la === 17 || _la === 18)) {
                                            localctx._op = this._errHandler.recoverInline(this);
                                        }
                                        else {
                                            this._errHandler.reportMatch(this);
                                            this.consume();
                                        }
                                        this.state = 345;
                                        this._errHandler.sync(this);
                                        _la = this._input.LA(1);
                                        if (_la === 14) {
                                            {
                                                this.state = 343;
                                                this.match(DaphneDSLGrammarParser.T__13);
                                                this.state = 344;
                                                localctx._kernel = this.match(DaphneDSLGrammarParser.IDENTIFIER);
                                            }
                                        }
                                        this.state = 347;
                                        localctx._rhs = this.expr(9);
                                    }
                                    break;
                                case 6:
                                    {
                                        localctx = new CmpExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 348;
                                        if (!(this.precpred(this._ctx, 7))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 7)");
                                        }
                                        this.state = 349;
                                        localctx._op = this._input.LT(1);
                                        _la = this._input.LA(1);
                                        if (!((((_la) & ~0x1F) === 0 && ((1 << _la) & 251664384) !== 0))) {
                                            localctx._op = this._errHandler.recoverInline(this);
                                        }
                                        else {
                                            this._errHandler.reportMatch(this);
                                            this.consume();
                                        }
                                        this.state = 350;
                                        localctx._rhs = this.expr(8);
                                    }
                                    break;
                                case 7:
                                    {
                                        localctx = new ConjExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 351;
                                        if (!(this.precpred(this._ctx, 6))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 6)");
                                        }
                                        this.state = 352;
                                        localctx._op = this.match(DaphneDSLGrammarParser.T__27);
                                        this.state = 353;
                                        localctx._rhs = this.expr(7);
                                    }
                                    break;
                                case 8:
                                    {
                                        localctx = new DisjExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._lhs = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 354;
                                        if (!(this.precpred(this._ctx, 5))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 5)");
                                        }
                                        this.state = 355;
                                        localctx._op = this.match(DaphneDSLGrammarParser.T__28);
                                        this.state = 356;
                                        localctx._rhs = this.expr(6);
                                    }
                                    break;
                                case 9:
                                    {
                                        localctx = new CondExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._cond = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 357;
                                        if (!(this.precpred(this._ctx, 4))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 4)");
                                        }
                                        this.state = 358;
                                        this.match(DaphneDSLGrammarParser.T__29);
                                        this.state = 359;
                                        localctx._thenExpr = this.expr(0);
                                        this.state = 360;
                                        this.match(DaphneDSLGrammarParser.T__8);
                                        this.state = 361;
                                        localctx._elseExpr = this.expr(5);
                                    }
                                    break;
                                case 10:
                                    {
                                        localctx = new RightIdxFilterExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._obj = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 363;
                                        if (!(this.precpred(this._ctx, 15))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 15)");
                                        }
                                        this.state = 364;
                                        this.match(DaphneDSLGrammarParser.T__14);
                                        this.state = 366;
                                        this._errHandler.sync(this);
                                        _la = this._input.LA(1);
                                        if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                            {
                                                this.state = 365;
                                                localctx._rows = this.expr(0);
                                            }
                                        }
                                        this.state = 368;
                                        this.match(DaphneDSLGrammarParser.T__4);
                                        this.state = 370;
                                        this._errHandler.sync(this);
                                        _la = this._input.LA(1);
                                        if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                            {
                                                this.state = 369;
                                                localctx._cols = this.expr(0);
                                            }
                                        }
                                        this.state = 372;
                                        this.match(DaphneDSLGrammarParser.T__15);
                                    }
                                    break;
                                case 11:
                                    {
                                        localctx = new RightIdxExtractExprContext(this, new ExprContext(this, _parentctx, _parentState));
                                        localctx._obj = _prevctx;
                                        this.pushNewRecursionContext(localctx, _startState, DaphneDSLGrammarParser.RULE_expr);
                                        this.state = 373;
                                        if (!(this.precpred(this._ctx, 14))) {
                                            throw this.createFailedPredicateException("this.precpred(this._ctx, 14)");
                                        }
                                        this.state = 374;
                                        localctx._idx = this.indexing();
                                    }
                                    break;
                            }
                        }
                    }
                    this.state = 379;
                    this._errHandler.sync(this);
                    _alt = this._interp.adaptivePredict(this._input, 44, this._ctx);
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    frameRow() {
        const localctx = new FrameRowContext(this, this._ctx, this.state);
        this.enterRule(localctx, 32, DaphneDSLGrammarParser.RULE_frameRow);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 380;
                this.match(DaphneDSLGrammarParser.T__30);
                this.state = 389;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                    {
                        this.state = 381;
                        this.expr(0);
                        this.state = 386;
                        this._errHandler.sync(this);
                        _la = this._input.LA(1);
                        while (_la === 5) {
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
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    indexing() {
        const localctx = new IndexingContext(this, this._ctx, this.state);
        this.enterRule(localctx, 34, DaphneDSLGrammarParser.RULE_indexing);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 393;
                this.match(DaphneDSLGrammarParser.T__30);
                this.state = 395;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885700) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                    {
                        this.state = 394;
                        localctx._rows = this.range();
                    }
                }
                this.state = 397;
                this.match(DaphneDSLGrammarParser.T__4);
                this.state = 399;
                this._errHandler.sync(this);
                _la = this._input.LA(1);
                if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885700) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                    {
                        this.state = 398;
                        localctx._cols = this.range();
                    }
                }
                this.state = 401;
                this.match(DaphneDSLGrammarParser.T__31);
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    range() {
        const localctx = new RangeContext(this, this._ctx, this.state);
        this.enterRule(localctx, 36, DaphneDSLGrammarParser.RULE_range);
        let _la;
        try {
            this.state = 411;
            this._errHandler.sync(this);
            switch (this._interp.adaptivePredict(this._input, 51, this._ctx)) {
                case 1:
                    this.enterOuterAlt(localctx, 1);
                    {
                        this.state = 403;
                        localctx._pos = this.expr(0);
                    }
                    break;
                case 2:
                    this.enterOuterAlt(localctx, 2);
                    {
                        {
                            this.state = 405;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                {
                                    this.state = 404;
                                    localctx._posLowerIncl = this.expr(0);
                                }
                            }
                            this.state = 407;
                            this.match(DaphneDSLGrammarParser.T__8);
                            this.state = 409;
                            this._errHandler.sync(this);
                            _la = this._input.LA(1);
                            if ((((_la) & ~0x1F) === 0 && ((1 << _la) & 2147885188) !== 0) || ((((_la - 39)) & ~0x1F) === 0 && ((1 << (_la - 39)) & 3847) !== 0)) {
                                {
                                    this.state = 408;
                                    localctx._posUpperExcl = this.expr(0);
                                }
                            }
                        }
                    }
                    break;
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    literal() {
        const localctx = new LiteralContext(this, this._ctx, this.state);
        this.enterRule(localctx, 38, DaphneDSLGrammarParser.RULE_literal);
        try {
            this.state = 417;
            this._errHandler.sync(this);
            switch (this._input.LA(1)) {
                case 47:
                    this.enterOuterAlt(localctx, 1);
                    {
                        this.state = 413;
                        this.match(DaphneDSLGrammarParser.INT_LITERAL);
                    }
                    break;
                case 48:
                    this.enterOuterAlt(localctx, 2);
                    {
                        this.state = 414;
                        this.match(DaphneDSLGrammarParser.FLOAT_LITERAL);
                    }
                    break;
                case 39:
                case 40:
                    this.enterOuterAlt(localctx, 3);
                    {
                        this.state = 415;
                        localctx._bl = this.boolLiteral();
                    }
                    break;
                case 49:
                    this.enterOuterAlt(localctx, 4);
                    {
                        this.state = 416;
                        this.match(DaphneDSLGrammarParser.STRING_LITERAL);
                    }
                    break;
                default:
                    throw new antlr4_1.NoViableAltException(this);
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    // @RuleVersion(0)
    boolLiteral() {
        const localctx = new BoolLiteralContext(this, this._ctx, this.state);
        this.enterRule(localctx, 40, DaphneDSLGrammarParser.RULE_boolLiteral);
        let _la;
        try {
            this.enterOuterAlt(localctx, 1);
            {
                this.state = 419;
                _la = this._input.LA(1);
                if (!(_la === 39 || _la === 40)) {
                    this._errHandler.recoverInline(this);
                }
                else {
                    this._errHandler.reportMatch(this);
                    this.consume();
                }
            }
        }
        catch (re) {
            if (re instanceof antlr4_1.RecognitionException) {
                localctx.exception = re;
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
        return localctx;
    }
    sempred(localctx, ruleIndex, predIndex) {
        switch (ruleIndex) {
            case 15:
                return this.expr_sempred(localctx, predIndex);
        }
        return true;
    }
    expr_sempred(localctx, predIndex) {
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
            DaphneDSLGrammarParser.__ATN = new antlr4_1.ATNDeserializer().deserialize(DaphneDSLGrammarParser._serializedATN);
        }
        return DaphneDSLGrammarParser.__ATN;
    }
}
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
DaphneDSLGrammarParser.EOF = antlr4_1.Token.EOF;
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
DaphneDSLGrammarParser.literalNames = [null, "';'",
    "'{'", "'}'",
    "'.'", "','",
    "'='", "'('",
    "')'", "':'",
    "'->'", "'<'",
    "'>'", "'$'",
    "'::'", "'[['",
    "']]'", "'+'",
    "'-'", "'@'",
    "'^'", "'%'",
    "'*'", "'/'",
    "'=='", "'!='",
    "'<='", "'>='",
    "'&&'", "'||'",
    "'?'", "'['",
    "']'", "'if'",
    "'else'", "'while'",
    "'do'", "'for'",
    "'in'", "'true'",
    "'false'", "'as'",
    "'def'", "'return'",
    "'import'"];
DaphneDSLGrammarParser.symbolicNames = [null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, null,
    null, "KW_IF",
    "KW_ELSE",
    "KW_WHILE",
    "KW_DO", "KW_FOR",
    "KW_IN", "KW_TRUE",
    "KW_FALSE",
    "KW_AS", "KW_DEF",
    "KW_RETURN",
    "KW_IMPORT",
    "DATA_TYPE",
    "VALUE_TYPE",
    "INT_LITERAL",
    "FLOAT_LITERAL",
    "STRING_LITERAL",
    "IDENTIFIER",
    "SCRIPT_STYLE_LINE_COMMENT",
    "C_STYLE_LINE_COMMENT",
    "MULTILINE_BLOCK_COMMENT",
    "WS"];
DaphneDSLGrammarParser.ruleNames = [
    "script", "statement", "importStatement", "blockStatement", "exprStatement",
    "assignStatement", "ifStatement", "whileStatement", "forStatement", "functionStatement",
    "returnStatement", "functionArgs", "functionArg", "functionRetTypes",
    "funcTypeDef", "expr", "frameRow", "indexing", "range", "literal", "boolLiteral",
];
DaphneDSLGrammarParser._serializedATN = [4, 1, 54, 422, 2, 0, 7, 0, 2,
    1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7, 4, 2, 5, 7, 5, 2, 6, 7, 6, 2, 7, 7, 7, 2, 8, 7, 8, 2, 9, 7, 9, 2,
    10, 7, 10, 2, 11, 7, 11, 2, 12, 7, 12, 2, 13, 7, 13, 2, 14, 7, 14, 2, 15, 7, 15, 2, 16, 7, 16, 2, 17,
    7, 17, 2, 18, 7, 18, 2, 19, 7, 19, 2, 20, 7, 20, 1, 0, 5, 0, 44, 8, 0, 10, 0, 12, 0, 47, 9, 0, 1, 0,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 60, 8, 1, 1, 2, 1, 2, 1, 2, 1, 2, 3, 2,
    66, 8, 2, 1, 2, 1, 2, 1, 3, 1, 3, 5, 3, 72, 8, 3, 10, 3, 12, 3, 75, 9, 3, 1, 3, 1, 3, 3, 3, 79, 8, 3, 1,
    4, 1, 4, 1, 4, 1, 5, 1, 5, 5, 5, 86, 8, 5, 10, 5, 12, 5, 89, 9, 5, 1, 5, 1, 5, 3, 5, 93, 8, 5, 1, 5, 1,
    5, 1, 5, 5, 5, 98, 8, 5, 10, 5, 12, 5, 101, 9, 5, 1, 5, 1, 5, 3, 5, 105, 8, 5, 5, 5, 107, 8, 5, 10, 5,
    12, 5, 110, 9, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 3, 6, 123, 8, 6, 1,
    7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 3, 7, 138, 8, 7, 3, 7, 140, 8,
    7, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 3, 8, 151, 8, 8, 1, 8, 1, 8, 1, 8, 1, 9, 1, 9, 1,
    9, 1, 9, 3, 9, 160, 8, 9, 1, 9, 1, 9, 1, 9, 3, 9, 165, 8, 9, 1, 9, 1, 9, 1, 10, 1, 10, 1, 10, 1, 10, 5,
    10, 173, 8, 10, 10, 10, 12, 10, 176, 9, 10, 3, 10, 178, 8, 10, 1, 10, 1, 10, 1, 11, 1, 11, 1, 11,
    5, 11, 185, 8, 11, 10, 11, 12, 11, 188, 9, 11, 1, 11, 3, 11, 191, 8, 11, 1, 12, 1, 12, 1, 12, 3,
    12, 196, 8, 12, 1, 13, 1, 13, 1, 13, 5, 13, 201, 8, 13, 10, 13, 12, 13, 204, 9, 13, 1, 14, 1, 14,
    1, 14, 1, 14, 3, 14, 210, 8, 14, 1, 14, 3, 14, 213, 8, 14, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15,
    5, 15, 221, 8, 15, 10, 15, 12, 15, 224, 9, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 5,
    15, 233, 8, 15, 10, 15, 12, 15, 236, 9, 15, 1, 15, 1, 15, 1, 15, 3, 15, 241, 8, 15, 1, 15, 1, 15,
    1, 15, 1, 15, 5, 15, 247, 8, 15, 10, 15, 12, 15, 250, 9, 15, 3, 15, 252, 8, 15, 1, 15, 1, 15, 1,
    15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 3, 15, 265, 8, 15, 1, 15, 1, 15, 1, 15,
    1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 5, 15, 277, 8, 15, 10, 15, 12, 15, 280, 9, 15, 3,
    15, 282, 8, 15, 1, 15, 1, 15, 1, 15, 3, 15, 287, 8, 15, 1, 15, 1, 15, 3, 15, 291, 8, 15, 1, 15, 3,
    15, 294, 8, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 5, 15, 305, 8, 15,
    10, 15, 12, 15, 308, 9, 15, 3, 15, 310, 8, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 5, 15, 317, 8,
    15, 10, 15, 12, 15, 320, 9, 15, 1, 15, 1, 15, 3, 15, 324, 8, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15,
    1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 3, 15, 339, 8, 15, 1, 15, 1, 15, 1, 15, 1,
    15, 1, 15, 3, 15, 346, 8, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15,
    1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 3, 15, 367, 8, 15, 1, 15, 1, 15, 3,
    15, 371, 8, 15, 1, 15, 1, 15, 1, 15, 5, 15, 376, 8, 15, 10, 15, 12, 15, 379, 9, 15, 1, 16, 1, 16,
    1, 16, 1, 16, 5, 16, 385, 8, 16, 10, 16, 12, 16, 388, 9, 16, 3, 16, 390, 8, 16, 1, 16, 1, 16, 1,
    17, 1, 17, 3, 17, 396, 8, 17, 1, 17, 1, 17, 3, 17, 400, 8, 17, 1, 17, 1, 17, 1, 18, 1, 18, 3, 18,
    406, 8, 18, 1, 18, 1, 18, 3, 18, 410, 8, 18, 3, 18, 412, 8, 18, 1, 19, 1, 19, 1, 19, 1, 19, 3, 19,
    418, 8, 19, 1, 20, 1, 20, 1, 20, 0, 1, 30, 21, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
    28, 30, 32, 34, 36, 38, 40, 0, 4, 1, 0, 17, 18, 1, 0, 22, 23, 2, 0, 11, 12, 24, 27, 1, 0, 39, 40,
    480, 0, 45, 1, 0, 0, 0, 2, 59, 1, 0, 0, 0, 4, 61, 1, 0, 0, 0, 6, 69, 1, 0, 0, 0, 8, 80, 1, 0, 0, 0, 10,
    87, 1, 0, 0, 0, 12, 115, 1, 0, 0, 0, 14, 139, 1, 0, 0, 0, 16, 141, 1, 0, 0, 0, 18, 155, 1, 0, 0, 0,
    20, 168, 1, 0, 0, 0, 22, 181, 1, 0, 0, 0, 24, 192, 1, 0, 0, 0, 26, 197, 1, 0, 0, 0, 28, 212, 1, 0,
    0, 0, 30, 323, 1, 0, 0, 0, 32, 380, 1, 0, 0, 0, 34, 393, 1, 0, 0, 0, 36, 411, 1, 0, 0, 0, 38, 417,
    1, 0, 0, 0, 40, 419, 1, 0, 0, 0, 42, 44, 3, 2, 1, 0, 43, 42, 1, 0, 0, 0, 44, 47, 1, 0, 0, 0, 45, 43,
    1, 0, 0, 0, 45, 46, 1, 0, 0, 0, 46, 48, 1, 0, 0, 0, 47, 45, 1, 0, 0, 0, 48, 49, 5, 0, 0, 1, 49, 1, 1,
    0, 0, 0, 50, 60, 3, 6, 3, 0, 51, 60, 3, 8, 4, 0, 52, 60, 3, 10, 5, 0, 53, 60, 3, 12, 6, 0, 54, 60, 3,
    14, 7, 0, 55, 60, 3, 16, 8, 0, 56, 60, 3, 18, 9, 0, 57, 60, 3, 20, 10, 0, 58, 60, 3, 4, 2, 0, 59, 50,
    1, 0, 0, 0, 59, 51, 1, 0, 0, 0, 59, 52, 1, 0, 0, 0, 59, 53, 1, 0, 0, 0, 59, 54, 1, 0, 0, 0, 59, 55, 1,
    0, 0, 0, 59, 56, 1, 0, 0, 0, 59, 57, 1, 0, 0, 0, 59, 58, 1, 0, 0, 0, 60, 3, 1, 0, 0, 0, 61, 62, 5, 44,
    0, 0, 62, 65, 5, 49, 0, 0, 63, 64, 5, 41, 0, 0, 64, 66, 5, 49, 0, 0, 65, 63, 1, 0, 0, 0, 65, 66, 1,
    0, 0, 0, 66, 67, 1, 0, 0, 0, 67, 68, 5, 1, 0, 0, 68, 5, 1, 0, 0, 0, 69, 73, 5, 2, 0, 0, 70, 72, 3, 2,
    1, 0, 71, 70, 1, 0, 0, 0, 72, 75, 1, 0, 0, 0, 73, 71, 1, 0, 0, 0, 73, 74, 1, 0, 0, 0, 74, 76, 1, 0, 0,
    0, 75, 73, 1, 0, 0, 0, 76, 78, 5, 3, 0, 0, 77, 79, 5, 1, 0, 0, 78, 77, 1, 0, 0, 0, 78, 79, 1, 0, 0, 0,
    79, 7, 1, 0, 0, 0, 80, 81, 3, 30, 15, 0, 81, 82, 5, 1, 0, 0, 82, 9, 1, 0, 0, 0, 83, 84, 5, 50, 0, 0,
    84, 86, 5, 4, 0, 0, 85, 83, 1, 0, 0, 0, 86, 89, 1, 0, 0, 0, 87, 85, 1, 0, 0, 0, 87, 88, 1, 0, 0, 0, 88,
    90, 1, 0, 0, 0, 89, 87, 1, 0, 0, 0, 90, 92, 5, 50, 0, 0, 91, 93, 3, 34, 17, 0, 92, 91, 1, 0, 0, 0, 92,
    93, 1, 0, 0, 0, 93, 108, 1, 0, 0, 0, 94, 99, 5, 5, 0, 0, 95, 96, 5, 50, 0, 0, 96, 98, 5, 4, 0, 0, 97,
    95, 1, 0, 0, 0, 98, 101, 1, 0, 0, 0, 99, 97, 1, 0, 0, 0, 99, 100, 1, 0, 0, 0, 100, 102, 1, 0, 0, 0,
    101, 99, 1, 0, 0, 0, 102, 104, 5, 50, 0, 0, 103, 105, 3, 34, 17, 0, 104, 103, 1, 0, 0, 0, 104, 105,
    1, 0, 0, 0, 105, 107, 1, 0, 0, 0, 106, 94, 1, 0, 0, 0, 107, 110, 1, 0, 0, 0, 108, 106, 1, 0, 0, 0,
    108, 109, 1, 0, 0, 0, 109, 111, 1, 0, 0, 0, 110, 108, 1, 0, 0, 0, 111, 112, 5, 6, 0, 0, 112, 113,
    3, 30, 15, 0, 113, 114, 5, 1, 0, 0, 114, 11, 1, 0, 0, 0, 115, 116, 5, 33, 0, 0, 116, 117, 5, 7, 0,
    0, 117, 118, 3, 30, 15, 0, 118, 119, 5, 8, 0, 0, 119, 122, 3, 2, 1, 0, 120, 121, 5, 34, 0, 0, 121,
    123, 3, 2, 1, 0, 122, 120, 1, 0, 0, 0, 122, 123, 1, 0, 0, 0, 123, 13, 1, 0, 0, 0, 124, 125, 5, 35,
    0, 0, 125, 126, 5, 7, 0, 0, 126, 127, 3, 30, 15, 0, 127, 128, 5, 8, 0, 0, 128, 129, 3, 2, 1, 0, 129,
    140, 1, 0, 0, 0, 130, 131, 5, 36, 0, 0, 131, 132, 3, 2, 1, 0, 132, 133, 5, 35, 0, 0, 133, 134, 5,
    7, 0, 0, 134, 135, 3, 30, 15, 0, 135, 137, 5, 8, 0, 0, 136, 138, 5, 1, 0, 0, 137, 136, 1, 0, 0, 0,
    137, 138, 1, 0, 0, 0, 138, 140, 1, 0, 0, 0, 139, 124, 1, 0, 0, 0, 139, 130, 1, 0, 0, 0, 140, 15,
    1, 0, 0, 0, 141, 142, 5, 37, 0, 0, 142, 143, 5, 7, 0, 0, 143, 144, 5, 50, 0, 0, 144, 145, 5, 38,
    0, 0, 145, 146, 3, 30, 15, 0, 146, 147, 5, 9, 0, 0, 147, 150, 3, 30, 15, 0, 148, 149, 5, 9, 0, 0,
    149, 151, 3, 30, 15, 0, 150, 148, 1, 0, 0, 0, 150, 151, 1, 0, 0, 0, 151, 152, 1, 0, 0, 0, 152, 153,
    5, 8, 0, 0, 153, 154, 3, 2, 1, 0, 154, 17, 1, 0, 0, 0, 155, 156, 5, 42, 0, 0, 156, 157, 5, 50, 0,
    0, 157, 159, 5, 7, 0, 0, 158, 160, 3, 22, 11, 0, 159, 158, 1, 0, 0, 0, 159, 160, 1, 0, 0, 0, 160,
    161, 1, 0, 0, 0, 161, 164, 5, 8, 0, 0, 162, 163, 5, 10, 0, 0, 163, 165, 3, 26, 13, 0, 164, 162,
    1, 0, 0, 0, 164, 165, 1, 0, 0, 0, 165, 166, 1, 0, 0, 0, 166, 167, 3, 6, 3, 0, 167, 19, 1, 0, 0, 0,
    168, 177, 5, 43, 0, 0, 169, 174, 3, 30, 15, 0, 170, 171, 5, 5, 0, 0, 171, 173, 3, 30, 15, 0, 172,
    170, 1, 0, 0, 0, 173, 176, 1, 0, 0, 0, 174, 172, 1, 0, 0, 0, 174, 175, 1, 0, 0, 0, 175, 178, 1, 0,
    0, 0, 176, 174, 1, 0, 0, 0, 177, 169, 1, 0, 0, 0, 177, 178, 1, 0, 0, 0, 178, 179, 1, 0, 0, 0, 179,
    180, 5, 1, 0, 0, 180, 21, 1, 0, 0, 0, 181, 186, 3, 24, 12, 0, 182, 183, 5, 5, 0, 0, 183, 185, 3,
    24, 12, 0, 184, 182, 1, 0, 0, 0, 185, 188, 1, 0, 0, 0, 186, 184, 1, 0, 0, 0, 186, 187, 1, 0, 0, 0,
    187, 190, 1, 0, 0, 0, 188, 186, 1, 0, 0, 0, 189, 191, 5, 5, 0, 0, 190, 189, 1, 0, 0, 0, 190, 191,
    1, 0, 0, 0, 191, 23, 1, 0, 0, 0, 192, 195, 5, 50, 0, 0, 193, 194, 5, 9, 0, 0, 194, 196, 3, 28, 14,
    0, 195, 193, 1, 0, 0, 0, 195, 196, 1, 0, 0, 0, 196, 25, 1, 0, 0, 0, 197, 202, 3, 28, 14, 0, 198,
    199, 5, 5, 0, 0, 199, 201, 3, 28, 14, 0, 200, 198, 1, 0, 0, 0, 201, 204, 1, 0, 0, 0, 202, 200, 1,
    0, 0, 0, 202, 203, 1, 0, 0, 0, 203, 27, 1, 0, 0, 0, 204, 202, 1, 0, 0, 0, 205, 209, 5, 45, 0, 0, 206,
    207, 5, 11, 0, 0, 207, 208, 5, 46, 0, 0, 208, 210, 5, 12, 0, 0, 209, 206, 1, 0, 0, 0, 209, 210,
    1, 0, 0, 0, 210, 213, 1, 0, 0, 0, 211, 213, 5, 46, 0, 0, 212, 205, 1, 0, 0, 0, 212, 211, 1, 0, 0,
    0, 213, 29, 1, 0, 0, 0, 214, 215, 6, 15, -1, 0, 215, 324, 3, 38, 19, 0, 216, 217, 5, 13, 0, 0, 217,
    324, 5, 50, 0, 0, 218, 219, 5, 50, 0, 0, 219, 221, 5, 4, 0, 0, 220, 218, 1, 0, 0, 0, 221, 224, 1,
    0, 0, 0, 222, 220, 1, 0, 0, 0, 222, 223, 1, 0, 0, 0, 223, 225, 1, 0, 0, 0, 224, 222, 1, 0, 0, 0, 225,
    324, 5, 50, 0, 0, 226, 227, 5, 7, 0, 0, 227, 228, 3, 30, 15, 0, 228, 229, 5, 8, 0, 0, 229, 324,
    1, 0, 0, 0, 230, 231, 5, 50, 0, 0, 231, 233, 5, 4, 0, 0, 232, 230, 1, 0, 0, 0, 233, 236, 1, 0, 0,
    0, 234, 232, 1, 0, 0, 0, 234, 235, 1, 0, 0, 0, 235, 237, 1, 0, 0, 0, 236, 234, 1, 0, 0, 0, 237, 240,
    5, 50, 0, 0, 238, 239, 5, 14, 0, 0, 239, 241, 5, 50, 0, 0, 240, 238, 1, 0, 0, 0, 240, 241, 1, 0,
    0, 0, 241, 242, 1, 0, 0, 0, 242, 251, 5, 7, 0, 0, 243, 248, 3, 30, 15, 0, 244, 245, 5, 5, 0, 0, 245,
    247, 3, 30, 15, 0, 246, 244, 1, 0, 0, 0, 247, 250, 1, 0, 0, 0, 248, 246, 1, 0, 0, 0, 248, 249, 1,
    0, 0, 0, 249, 252, 1, 0, 0, 0, 250, 248, 1, 0, 0, 0, 251, 243, 1, 0, 0, 0, 251, 252, 1, 0, 0, 0, 252,
    253, 1, 0, 0, 0, 253, 324, 5, 8, 0, 0, 254, 264, 5, 41, 0, 0, 255, 256, 5, 4, 0, 0, 256, 265, 5,
    45, 0, 0, 257, 258, 5, 4, 0, 0, 258, 265, 5, 46, 0, 0, 259, 260, 5, 4, 0, 0, 260, 261, 5, 45, 0,
    0, 261, 262, 5, 11, 0, 0, 262, 263, 5, 46, 0, 0, 263, 265, 5, 12, 0, 0, 264, 255, 1, 0, 0, 0, 264,
    257, 1, 0, 0, 0, 264, 259, 1, 0, 0, 0, 265, 266, 1, 0, 0, 0, 266, 267, 5, 7, 0, 0, 267, 268, 3, 30,
    15, 0, 268, 269, 5, 8, 0, 0, 269, 324, 1, 0, 0, 0, 270, 271, 7, 0, 0, 0, 271, 324, 3, 30, 15, 13,
    272, 281, 5, 31, 0, 0, 273, 278, 3, 30, 15, 0, 274, 275, 5, 5, 0, 0, 275, 277, 3, 30, 15, 0, 276,
    274, 1, 0, 0, 0, 277, 280, 1, 0, 0, 0, 278, 276, 1, 0, 0, 0, 278, 279, 1, 0, 0, 0, 279, 282, 1, 0,
    0, 0, 280, 278, 1, 0, 0, 0, 281, 273, 1, 0, 0, 0, 281, 282, 1, 0, 0, 0, 282, 283, 1, 0, 0, 0, 283,
    293, 5, 32, 0, 0, 284, 286, 5, 7, 0, 0, 285, 287, 3, 30, 15, 0, 286, 285, 1, 0, 0, 0, 286, 287,
    1, 0, 0, 0, 287, 288, 1, 0, 0, 0, 288, 290, 5, 5, 0, 0, 289, 291, 3, 30, 15, 0, 290, 289, 1, 0, 0,
    0, 290, 291, 1, 0, 0, 0, 291, 292, 1, 0, 0, 0, 292, 294, 5, 8, 0, 0, 293, 284, 1, 0, 0, 0, 293, 294,
    1, 0, 0, 0, 294, 324, 1, 0, 0, 0, 295, 309, 5, 2, 0, 0, 296, 297, 3, 30, 15, 0, 297, 298, 5, 9, 0,
    0, 298, 306, 3, 30, 15, 0, 299, 300, 5, 5, 0, 0, 300, 301, 3, 30, 15, 0, 301, 302, 5, 9, 0, 0, 302,
    303, 3, 30, 15, 0, 303, 305, 1, 0, 0, 0, 304, 299, 1, 0, 0, 0, 305, 308, 1, 0, 0, 0, 306, 304, 1,
    0, 0, 0, 306, 307, 1, 0, 0, 0, 307, 310, 1, 0, 0, 0, 308, 306, 1, 0, 0, 0, 309, 296, 1, 0, 0, 0, 309,
    310, 1, 0, 0, 0, 310, 311, 1, 0, 0, 0, 311, 324, 5, 3, 0, 0, 312, 313, 5, 2, 0, 0, 313, 318, 3, 32,
    16, 0, 314, 315, 5, 5, 0, 0, 315, 317, 3, 32, 16, 0, 316, 314, 1, 0, 0, 0, 317, 320, 1, 0, 0, 0,
    318, 316, 1, 0, 0, 0, 318, 319, 1, 0, 0, 0, 319, 321, 1, 0, 0, 0, 320, 318, 1, 0, 0, 0, 321, 322,
    5, 3, 0, 0, 322, 324, 1, 0, 0, 0, 323, 214, 1, 0, 0, 0, 323, 216, 1, 0, 0, 0, 323, 222, 1, 0, 0, 0,
    323, 226, 1, 0, 0, 0, 323, 234, 1, 0, 0, 0, 323, 254, 1, 0, 0, 0, 323, 270, 1, 0, 0, 0, 323, 272,
    1, 0, 0, 0, 323, 295, 1, 0, 0, 0, 323, 312, 1, 0, 0, 0, 324, 377, 1, 0, 0, 0, 325, 326, 10, 12, 0,
    0, 326, 327, 5, 19, 0, 0, 327, 376, 3, 30, 15, 13, 328, 329, 10, 11, 0, 0, 329, 330, 5, 20, 0,
    0, 330, 376, 3, 30, 15, 12, 331, 332, 10, 10, 0, 0, 332, 333, 5, 21, 0, 0, 333, 376, 3, 30, 15,
    11, 334, 335, 10, 9, 0, 0, 335, 338, 7, 1, 0, 0, 336, 337, 5, 14, 0, 0, 337, 339, 5, 50, 0, 0, 338,
    336, 1, 0, 0, 0, 338, 339, 1, 0, 0, 0, 339, 340, 1, 0, 0, 0, 340, 376, 3, 30, 15, 10, 341, 342,
    10, 8, 0, 0, 342, 345, 7, 0, 0, 0, 343, 344, 5, 14, 0, 0, 344, 346, 5, 50, 0, 0, 345, 343, 1, 0,
    0, 0, 345, 346, 1, 0, 0, 0, 346, 347, 1, 0, 0, 0, 347, 376, 3, 30, 15, 9, 348, 349, 10, 7, 0, 0,
    349, 350, 7, 2, 0, 0, 350, 376, 3, 30, 15, 8, 351, 352, 10, 6, 0, 0, 352, 353, 5, 28, 0, 0, 353,
    376, 3, 30, 15, 7, 354, 355, 10, 5, 0, 0, 355, 356, 5, 29, 0, 0, 356, 376, 3, 30, 15, 6, 357, 358,
    10, 4, 0, 0, 358, 359, 5, 30, 0, 0, 359, 360, 3, 30, 15, 0, 360, 361, 5, 9, 0, 0, 361, 362, 3, 30,
    15, 5, 362, 376, 1, 0, 0, 0, 363, 364, 10, 15, 0, 0, 364, 366, 5, 15, 0, 0, 365, 367, 3, 30, 15,
    0, 366, 365, 1, 0, 0, 0, 366, 367, 1, 0, 0, 0, 367, 368, 1, 0, 0, 0, 368, 370, 5, 5, 0, 0, 369, 371,
    3, 30, 15, 0, 370, 369, 1, 0, 0, 0, 370, 371, 1, 0, 0, 0, 371, 372, 1, 0, 0, 0, 372, 376, 5, 16,
    0, 0, 373, 374, 10, 14, 0, 0, 374, 376, 3, 34, 17, 0, 375, 325, 1, 0, 0, 0, 375, 328, 1, 0, 0, 0,
    375, 331, 1, 0, 0, 0, 375, 334, 1, 0, 0, 0, 375, 341, 1, 0, 0, 0, 375, 348, 1, 0, 0, 0, 375, 351,
    1, 0, 0, 0, 375, 354, 1, 0, 0, 0, 375, 357, 1, 0, 0, 0, 375, 363, 1, 0, 0, 0, 375, 373, 1, 0, 0, 0,
    376, 379, 1, 0, 0, 0, 377, 375, 1, 0, 0, 0, 377, 378, 1, 0, 0, 0, 378, 31, 1, 0, 0, 0, 379, 377,
    1, 0, 0, 0, 380, 389, 5, 31, 0, 0, 381, 386, 3, 30, 15, 0, 382, 383, 5, 5, 0, 0, 383, 385, 3, 30,
    15, 0, 384, 382, 1, 0, 0, 0, 385, 388, 1, 0, 0, 0, 386, 384, 1, 0, 0, 0, 386, 387, 1, 0, 0, 0, 387,
    390, 1, 0, 0, 0, 388, 386, 1, 0, 0, 0, 389, 381, 1, 0, 0, 0, 389, 390, 1, 0, 0, 0, 390, 391, 1, 0,
    0, 0, 391, 392, 5, 32, 0, 0, 392, 33, 1, 0, 0, 0, 393, 395, 5, 31, 0, 0, 394, 396, 3, 36, 18, 0,
    395, 394, 1, 0, 0, 0, 395, 396, 1, 0, 0, 0, 396, 397, 1, 0, 0, 0, 397, 399, 5, 5, 0, 0, 398, 400,
    3, 36, 18, 0, 399, 398, 1, 0, 0, 0, 399, 400, 1, 0, 0, 0, 400, 401, 1, 0, 0, 0, 401, 402, 5, 32,
    0, 0, 402, 35, 1, 0, 0, 0, 403, 412, 3, 30, 15, 0, 404, 406, 3, 30, 15, 0, 405, 404, 1, 0, 0, 0,
    405, 406, 1, 0, 0, 0, 406, 407, 1, 0, 0, 0, 407, 409, 5, 9, 0, 0, 408, 410, 3, 30, 15, 0, 409, 408,
    1, 0, 0, 0, 409, 410, 1, 0, 0, 0, 410, 412, 1, 0, 0, 0, 411, 403, 1, 0, 0, 0, 411, 405, 1, 0, 0, 0,
    412, 37, 1, 0, 0, 0, 413, 418, 5, 47, 0, 0, 414, 418, 5, 48, 0, 0, 415, 418, 3, 40, 20, 0, 416,
    418, 5, 49, 0, 0, 417, 413, 1, 0, 0, 0, 417, 414, 1, 0, 0, 0, 417, 415, 1, 0, 0, 0, 417, 416, 1,
    0, 0, 0, 418, 39, 1, 0, 0, 0, 419, 420, 7, 3, 0, 0, 420, 41, 1, 0, 0, 0, 53, 45, 59, 65, 73, 78, 87,
    92, 99, 104, 108, 122, 137, 139, 150, 159, 164, 174, 177, 186, 190, 195, 202, 209, 212, 222,
    234, 240, 248, 251, 264, 278, 281, 286, 290, 293, 306, 309, 318, 323, 338, 345, 366, 370,
    375, 377, 386, 389, 395, 399, 405, 409, 411, 417];
DaphneDSLGrammarParser.DecisionsToDFA = DaphneDSLGrammarParser._ATN.decisionToState.map((ds, index) => new antlr4_1.DFA(ds, index));
exports.default = DaphneDSLGrammarParser;
class ScriptContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    EOF() {
        return this.getToken(DaphneDSLGrammarParser.EOF, 0);
    }
    statement_list() {
        return this.getTypedRuleContexts(StatementContext);
    }
    statement(i) {
        return this.getTypedRuleContext(StatementContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_script;
    }
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
class StatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    blockStatement() {
        return this.getTypedRuleContext(BlockStatementContext, 0);
    }
    exprStatement() {
        return this.getTypedRuleContext(ExprStatementContext, 0);
    }
    assignStatement() {
        return this.getTypedRuleContext(AssignStatementContext, 0);
    }
    ifStatement() {
        return this.getTypedRuleContext(IfStatementContext, 0);
    }
    whileStatement() {
        return this.getTypedRuleContext(WhileStatementContext, 0);
    }
    forStatement() {
        return this.getTypedRuleContext(ForStatementContext, 0);
    }
    functionStatement() {
        return this.getTypedRuleContext(FunctionStatementContext, 0);
    }
    returnStatement() {
        return this.getTypedRuleContext(ReturnStatementContext, 0);
    }
    importStatement() {
        return this.getTypedRuleContext(ImportStatementContext, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_statement;
    }
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
class ImportStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    KW_IMPORT() {
        return this.getToken(DaphneDSLGrammarParser.KW_IMPORT, 0);
    }
    STRING_LITERAL_list() {
        return this.getTokens(DaphneDSLGrammarParser.STRING_LITERAL);
    }
    STRING_LITERAL(i) {
        return this.getToken(DaphneDSLGrammarParser.STRING_LITERAL, i);
    }
    KW_AS() {
        return this.getToken(DaphneDSLGrammarParser.KW_AS, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_importStatement;
    }
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
class BlockStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    statement_list() {
        return this.getTypedRuleContexts(StatementContext);
    }
    statement(i) {
        return this.getTypedRuleContext(StatementContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_blockStatement;
    }
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
class ExprStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_exprStatement;
    }
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
class AssignStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    IDENTIFIER_list() {
        return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
    }
    IDENTIFIER(i) {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
    }
    indexing_list() {
        return this.getTypedRuleContexts(IndexingContext);
    }
    indexing(i) {
        return this.getTypedRuleContext(IndexingContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_assignStatement;
    }
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
class IfStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    KW_IF() {
        return this.getToken(DaphneDSLGrammarParser.KW_IF, 0);
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
    }
    statement_list() {
        return this.getTypedRuleContexts(StatementContext);
    }
    statement(i) {
        return this.getTypedRuleContext(StatementContext, i);
    }
    KW_ELSE() {
        return this.getToken(DaphneDSLGrammarParser.KW_ELSE, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_ifStatement;
    }
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
class WhileStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    KW_WHILE() {
        return this.getToken(DaphneDSLGrammarParser.KW_WHILE, 0);
    }
    KW_DO() {
        return this.getToken(DaphneDSLGrammarParser.KW_DO, 0);
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
    }
    statement() {
        return this.getTypedRuleContext(StatementContext, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_whileStatement;
    }
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
class ForStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    KW_FOR() {
        return this.getToken(DaphneDSLGrammarParser.KW_FOR, 0);
    }
    KW_IN() {
        return this.getToken(DaphneDSLGrammarParser.KW_IN, 0);
    }
    IDENTIFIER() {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
    }
    statement() {
        return this.getTypedRuleContext(StatementContext, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_forStatement;
    }
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
class FunctionStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    KW_DEF() {
        return this.getToken(DaphneDSLGrammarParser.KW_DEF, 0);
    }
    IDENTIFIER() {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0);
    }
    blockStatement() {
        return this.getTypedRuleContext(BlockStatementContext, 0);
    }
    functionArgs() {
        return this.getTypedRuleContext(FunctionArgsContext, 0);
    }
    functionRetTypes() {
        return this.getTypedRuleContext(FunctionRetTypesContext, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_functionStatement;
    }
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
class ReturnStatementContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    KW_RETURN() {
        return this.getToken(DaphneDSLGrammarParser.KW_RETURN, 0);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_returnStatement;
    }
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
class FunctionArgsContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    functionArg_list() {
        return this.getTypedRuleContexts(FunctionArgContext);
    }
    functionArg(i) {
        return this.getTypedRuleContext(FunctionArgContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_functionArgs;
    }
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
class FunctionArgContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    IDENTIFIER() {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0);
    }
    funcTypeDef() {
        return this.getTypedRuleContext(FuncTypeDefContext, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_functionArg;
    }
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
class FunctionRetTypesContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    funcTypeDef_list() {
        return this.getTypedRuleContexts(FuncTypeDefContext);
    }
    funcTypeDef(i) {
        return this.getTypedRuleContext(FuncTypeDefContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_functionRetTypes;
    }
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
class FuncTypeDefContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    DATA_TYPE() {
        return this.getToken(DaphneDSLGrammarParser.DATA_TYPE, 0);
    }
    VALUE_TYPE() {
        return this.getToken(DaphneDSLGrammarParser.VALUE_TYPE, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_funcTypeDef;
    }
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
class ExprContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_expr;
    }
    copyFrom(ctx) {
        super.copyFrom(ctx);
    }
}
exports.ExprContext = ExprContext;
class RightIdxExtractExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
    }
    indexing() {
        return this.getTypedRuleContext(IndexingContext, 0);
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
class ModExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class CastExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    KW_AS() {
        return this.getToken(DaphneDSLGrammarParser.KW_AS, 0);
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
    }
    DATA_TYPE() {
        return this.getToken(DaphneDSLGrammarParser.DATA_TYPE, 0);
    }
    VALUE_TYPE() {
        return this.getToken(DaphneDSLGrammarParser.VALUE_TYPE, 0);
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
class MatmulExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class CondExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class ConjExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class RightIdxFilterExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class MatrixLiteralExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class MinusExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
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
class ParanthesesExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr() {
        return this.getTypedRuleContext(ExprContext, 0);
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
class CmpExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class AddExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
    }
    IDENTIFIER() {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0);
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
class LiteralExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    literal() {
        return this.getTypedRuleContext(LiteralContext, 0);
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
class RowMajorFrameLiteralExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        this._rows = [];
        super.copyFrom(ctx);
    }
    frameRow_list() {
        return this.getTypedRuleContexts(FrameRowContext);
    }
    frameRow(i) {
        return this.getTypedRuleContext(FrameRowContext, i);
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
class MulExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
    }
    IDENTIFIER() {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0);
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
class ArgExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    IDENTIFIER() {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, 0);
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
class ColMajorFrameLiteralExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        this._labels = [];
        this._cols = [];
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class CallExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    IDENTIFIER_list() {
        return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
    }
    IDENTIFIER(i) {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class PowExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
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
class IdentifierExprContext extends ExprContext {
    constructor(parser, ctx) {
        super(parser, ctx.parentCtx, ctx.invokingState);
        super.copyFrom(ctx);
    }
    IDENTIFIER_list() {
        return this.getTokens(DaphneDSLGrammarParser.IDENTIFIER);
    }
    IDENTIFIER(i) {
        return this.getToken(DaphneDSLGrammarParser.IDENTIFIER, i);
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
class FrameRowContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_frameRow;
    }
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
class IndexingContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    range_list() {
        return this.getTypedRuleContexts(RangeContext);
    }
    range(i) {
        return this.getTypedRuleContext(RangeContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_indexing;
    }
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
class RangeContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    expr_list() {
        return this.getTypedRuleContexts(ExprContext);
    }
    expr(i) {
        return this.getTypedRuleContext(ExprContext, i);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_range;
    }
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
class LiteralContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    INT_LITERAL() {
        return this.getToken(DaphneDSLGrammarParser.INT_LITERAL, 0);
    }
    FLOAT_LITERAL() {
        return this.getToken(DaphneDSLGrammarParser.FLOAT_LITERAL, 0);
    }
    boolLiteral() {
        return this.getTypedRuleContext(BoolLiteralContext, 0);
    }
    STRING_LITERAL() {
        return this.getToken(DaphneDSLGrammarParser.STRING_LITERAL, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_literal;
    }
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
class BoolLiteralContext extends antlr4_1.ParserRuleContext {
    constructor(parser, parent, invokingState) {
        super(parent, invokingState);
        this.parser = parser;
    }
    KW_TRUE() {
        return this.getToken(DaphneDSLGrammarParser.KW_TRUE, 0);
    }
    KW_FALSE() {
        return this.getToken(DaphneDSLGrammarParser.KW_FALSE, 0);
    }
    get ruleIndex() {
        return DaphneDSLGrammarParser.RULE_boolLiteral;
    }
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