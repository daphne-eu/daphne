"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const node_1 = require("vscode-languageserver/node");
const vscode_languageserver_textdocument_1 = require("vscode-languageserver-textdocument");
const perf_hooks_1 = require("perf_hooks");
const DaphneDSLGrammarLexer_1 = require("./server/DaphneDSLGrammarLexer");
const DaphneDSLGrammarParser_1 = require("./server/DaphneDSLGrammarParser");
const antlr4ts_1 = require("antlr4ts");
const SemanticAnalyzer_1 = require("./server/SemanticAnalyzer");
const connection = (0, node_1.createConnection)(node_1.ProposedFeatures.all);
connection.console.log("✅ Daphne LSP server started!");
const documents = new node_1.TextDocuments(vscode_languageserver_textdocument_1.TextDocument);
connection.onInitialize((_params) => {
    return {
        capabilities: {
            textDocumentSync: node_1.TextDocumentSyncKind.Incremental,
            completionProvider: { resolveProvider: true },
            hoverProvider: true,
            definitionProvider: true,
            diagnosticProvider: {
                interFileDependencies: false,
                workspaceDiagnostics: false
            }
        }
    };
});
// 🔔 Syntax error listener:
class SyntaxErrorListener {
    constructor(document) {
        this.diagnostics = [];
        this.document = document;
    }
    syntaxError(_recognizer, offendingSymbol, line, charPositionInLine, msg, _e) {
        const start = this.document.offsetAt({ line: line - 1, character: charPositionInLine });
        const end = start + (offendingSymbol?.text?.length || 1);
        this.diagnostics.push({
            severity: node_1.DiagnosticSeverity.Error,
            range: {
                start: this.document.positionAt(start),
                end: this.document.positionAt(end)
            },
            message: `Syntax error: ${msg}`,
            source: 'daphne-lsp'
        });
    }
    getDiagnostics() {
        return this.diagnostics;
    }
}
async function validateTextDocument(textDocument) {
    const diagnostics = [];
    try {
        const inputStream = new antlr4ts_1.ANTLRInputStream(textDocument.getText());
        const lexer = new DaphneDSLGrammarLexer_1.DaphneDSLGrammarLexer(inputStream);
        const tokenStream = new antlr4ts_1.CommonTokenStream(lexer);
        const parser = new DaphneDSLGrammarParser_1.DaphneDSLGrammarParser(tokenStream);
        const syntaxListener = new SyntaxErrorListener(textDocument);
        parser.removeErrorListeners();
        parser.addErrorListener(syntaxListener);
        const tree = parser.script();
        diagnostics.push(...syntaxListener.getDiagnostics());
        // 🔔 Semantic analysis correctly hooked:
        const semanticAnalyzer = new SemanticAnalyzer_1.SemanticAnalyzer();
        const semanticDiagnostics = semanticAnalyzer.analyze(tree);
        diagnostics.push(...semanticDiagnostics);
    }
    catch (err) {
        diagnostics.push({
            severity: node_1.DiagnosticSeverity.Error,
            range: {
                start: textDocument.positionAt(0),
                end: textDocument.positionAt(1)
            },
            message: `Unexpected parse error: ${err.message}`,
            source: 'daphne-lsp'
        });
    }
    return diagnostics;
}
connection.languages.diagnostics.on(async (params) => {
    const document = documents.get(params.textDocument.uri);
    const start = perf_hooks_1.performance.now();
    if (!document) {
        return { kind: node_1.DocumentDiagnosticReportKind.Full, items: [] };
    }
    const diagnostics = await validateTextDocument(document);
    const end = perf_hooks_1.performance.now();
    connection.console.log(`🕒 Diagnostics for ${document.uri} took ${(end - start).toFixed(2)} ms`);
    return { kind: node_1.DocumentDiagnosticReportKind.Full, items: diagnostics };
});
connection.onCompletion((_params) => {
    const keywords = ['def', 'import', 'let', 'if', 'else', 'for', 'while', 'match', 'return', 'true', 'false', 'null'];
    return keywords.map((kw, index) => ({
        label: kw,
        kind: node_1.CompletionItemKind.Keyword,
        data: index
    }));
});
connection.onCompletionResolve((item) => {
    item.detail = 'Daphne DSL keyword';
    item.documentation = `Keyword \`${item.label}\``;
    return item;
});
connection.onHover((params) => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return null;
    }
    const position = params.position;
    const text = document.getText();
    const offset = document.offsetAt(position);
    const hoverWordMatch = /\b\w+\b/g;
    let word = null;
    let match;
    while ((match = hoverWordMatch.exec(text))) {
        if (offset >= match.index && offset <= match.index + match[0].length) {
            word = match[0];
            break;
        }
    }
    if (word) {
        return { contents: { kind: 'markdown', value: `Information about \`${word}\`` } };
    }
    return null;
});
connection.onDefinition((params) => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return [];
    }
    try {
        const text = document.getText();
        const offset = document.offsetAt(params.position);
        const wordMatch = /\b\w+\b/g;
        let word = null;
        let match;
        while ((match = wordMatch.exec(text))) {
            if (offset >= match.index && offset <= match.index + match[0].length) {
                word = match[0];
                break;
            }
        }
        if (!word) {
            return [];
        }
        const inputStream = new antlr4ts_1.ANTLRInputStream(text);
        const lexer = new DaphneDSLGrammarLexer_1.DaphneDSLGrammarLexer(inputStream);
        const tokenStream = new antlr4ts_1.CommonTokenStream(lexer);
        const parser = new DaphneDSLGrammarParser_1.DaphneDSLGrammarParser(tokenStream);
        const tree = parser.script(); // ❗ can throw
        const analyzer = new SemanticAnalyzer_1.SemanticAnalyzer();
        analyzer.analyze(tree);
        const symbols = analyzer.getSymbols();
        connection.console.log(`🔎 Symbol Table: ${JSON.stringify(Array.from(symbols.entries()))}`);
        connection.console.log(`🧠 Looking for word: ${word}`);
        const line = symbols.get(word);
        if (line === undefined) {
            return [];
        }
        return [{
                uri: params.textDocument.uri,
                range: {
                    start: { line, character: 0 },
                    end: { line, character: 100 }
                }
            }];
    }
    catch (err) {
        connection.console.error(`💥 Go-to Definition failed: ${err}`);
        return [];
    }
});
documents.listen(connection);
connection.listen();
//# sourceMappingURL=server.js.map