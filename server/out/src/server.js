"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const node_1 = require("vscode-languageserver/node");
const vscode_languageserver_textdocument_1 = require("vscode-languageserver-textdocument");
const perf_hooks_1 = require("perf_hooks");
// 🧩 ANTLR integration:
const antlr4ts_1 = require("antlr4ts");
const DaphneDSLGrammarLexer_1 = require("./DaphneDSLGrammarLexer");
const DaphneDSLGrammarParser_1 = require("../DaphneDSLGrammarParser");
const BailErrorStrategy_1 = require("antlr4ts/atn/BailErrorStrategy");
const RecognitionException_1 = require("antlr4ts/RecognitionException");
const connection = (0, node_1.createConnection)(node_1.ProposedFeatures.all);
connection.console.log("✅ Daphne LSP server started!");
const documents = new node_1.TextDocuments(vscode_languageserver_textdocument_1.TextDocument);
connection.onInitialize((params) => {
    const result = {
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
    return result;
});
// Diagnostics handler:
async function validateTextDocument(textDocument) {
    const text = textDocument.getText();
    const diagnostics = [];
    try {
        const inputStream = new antlr4ts_1.ANTLRInputStream(text);
        const lexer = new DaphneDSLGrammarLexer_1.DaphneDSLGrammarLexer(inputStream);
        const tokenStream = new antlr4ts_1.CommonTokenStream(lexer);
        const parser = new DaphneDSLGrammarParser_1.DaphneDSLGrammarParser(tokenStream);
        parser.errorHandler = new BailErrorStrategy_1.BailErrorStrategy();
        parser.script(); // 'script' is the top-level rule of Daphne grammar
    }
    catch (err) {
        if (err instanceof RecognitionException_1.RecognitionException) {
            const offendingToken = err.offendingToken;
            diagnostics.push({
                severity: node_1.DiagnosticSeverity.Error,
                range: {
                    start: textDocument.positionAt(offendingToken.startIndex),
                    end: textDocument.positionAt(offendingToken.stopIndex + 1)
                },
                message: `Syntax error: ${err.message}`,
                source: 'daphne-lsp'
            });
        }
        else {
            connection.console.error(`Unexpected parse error: ${err}`);
        }
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
// Basic completions (example using real Daphne keywords):
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
// Hover (basic example):
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
// Dummy go-to-definition (for now)
connection.onDefinition((_params) => {
    return [];
});
documents.listen(connection);
connection.listen();
//# sourceMappingURL=server.js.map