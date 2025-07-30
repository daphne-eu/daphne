"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SemanticAnalyzer = void 0;
const AbstractParseTreeVisitor_1 = require("antlr4ts/tree/AbstractParseTreeVisitor");
const vscode_languageserver_1 = require("vscode-languageserver");
const VALID_KEYWORDS = new Set([
    'if', 'else', 'while', 'do', 'for', 'in', 'true', 'false',
    'as', 'def', 'return', 'import'
]);
class SemanticAnalyzer extends AbstractParseTreeVisitor_1.AbstractParseTreeVisitor {
    constructor() {
        super(...arguments);
        this.declaredVars = new Map();
        this.diagnostics = [];
        this.symbolTable = new Map();
        this.currentFunctionReturnType = null;
    }
    defaultResult() {
        return;
    }
    // Collect all diagnostics, especially for unused vars
    getDiagnostics() {
        console.log("🔔 Collecting semantic diagnostics...");
        for (const [name, info] of this.declaredVars.entries()) {
            console.log(`  ➜ ${name} used=${info.used}`);
            if (!info.used) {
                this.diagnostics.push({
                    severity: vscode_languageserver_1.DiagnosticSeverity.Warning,
                    range: {
                        start: { line: info.line, character: 0 },
                        end: { line: info.line, character: 100 },
                    },
                    message: `⚠️ Variable '${name}' declared but never used.`,
                    source: 'semantic-check',
                });
            }
        }
        return this.diagnostics;
    }
    visitScript(ctx) {
        console.log("📜 visitScript triggered");
        this.visitChildren(ctx);
    }
    visitStatement(ctx) {
        console.log("🧱 visitStatement triggered");
        this.visitChildren(ctx);
    }
    visitAssignStatement(ctx) {
        console.log("📌 visitAssignStatement triggered");
        const children = ctx.children;
        if (children && children.length >= 3) {
            const varToken = children[0];
            const varName = typeof varToken.getText === 'function' ? varToken.getText() : 'unknown';
            const line = varToken.symbol?.line ?? 0;
            this.declaredVars.set(varName, { line: line - 1, used: false });
            this.symbolTable.set(varName, line - 1);
        }
        this.visitChildren(ctx);
    }
    visitVariableAccess(ctx) {
        const varName = ctx.text;
        const info = this.declaredVars.get(varName);
        if (info) {
            info.used = true;
        }
        this.visitChildren(ctx);
    }
    visitFunctionRetTypes(ctx) {
        const typeText = ctx.text;
        console.log(`📘 Return type extracted: ${typeText}`);
        this.currentFunctionReturnType = typeText;
        return this.visitChildren(ctx);
    }
    analyze(tree) {
        this.visit(tree);
        return this.getDiagnostics();
    }
    getSymbols() {
        return this.symbolTable;
    }
    inferTypeFromText(text) {
        if (/^\d+$/.test(text)) {
            return 'int';
        }
        if (/^".*"$/.test(text)) {
            return 'string';
        }
        if (/^\d+\.\d+$/.test(text)) {
            return 'float';
        }
        if (text === 'true' || text === 'false') {
            return 'bool';
        }
        return 'unknown';
    }
}
exports.SemanticAnalyzer = SemanticAnalyzer;
//# sourceMappingURL=SemanticAnalyzer.js.map