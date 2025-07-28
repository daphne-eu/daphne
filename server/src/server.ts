import {
  createConnection,
  TextDocuments,
  Diagnostic,
  DiagnosticSeverity,
  ProposedFeatures,
  InitializeParams,
  TextDocumentSyncKind,
  InitializeResult,
  DocumentDiagnosticReportKind,
  type DocumentDiagnosticReport,
  type DefinitionParams,
  type Location,
  type Hover,
  CompletionItem,
  CompletionItemKind,
  TextDocumentPositionParams
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';
import { performance } from 'perf_hooks';

import { DaphneDSLGrammarLexer } from './server/DaphneDSLGrammarLexer';
import { DaphneDSLGrammarParser } from './server/DaphneDSLGrammarParser';
import { ANTLRInputStream, CommonTokenStream } from 'antlr4ts';
import { ANTLRErrorListener, Recognizer, RecognitionException } from 'antlr4ts';

import { SemanticAnalyzer } from './server/SemanticAnalyzer';

const connection = createConnection(ProposedFeatures.all);
connection.console.log("✅ Daphne LSP server started!");

const documents = new TextDocuments(TextDocument);

connection.onInitialize((_params: InitializeParams) => {
  return {
    capabilities: {
      textDocumentSync: TextDocumentSyncKind.Incremental,
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
class SyntaxErrorListener implements ANTLRErrorListener<any> {
  private diagnostics: Diagnostic[] = [];
  private document: TextDocument;

  constructor(document: TextDocument) {
    this.document = document;
  }

  syntaxError(
    _recognizer: Recognizer<any, any>,
    offendingSymbol: any,
    line: number,
    charPositionInLine: number,
    msg: string,
    _e: RecognitionException | undefined
  ): void {
    const start = this.document.offsetAt({ line: line - 1, character: charPositionInLine });
    const end = start + (offendingSymbol?.text?.length || 1);

    this.diagnostics.push({
      severity: DiagnosticSeverity.Error,
      range: {
        start: this.document.positionAt(start),
        end: this.document.positionAt(end)
      },
      message: `Syntax error: ${msg}`,
      source: 'daphne-lsp'
    });
  }

  getDiagnostics(): Diagnostic[] {
    return this.diagnostics;
  }
}

async function validateTextDocument(textDocument: TextDocument): Promise<Diagnostic[]> {
  const diagnostics: Diagnostic[] = [];

  try {
    const inputStream = new ANTLRInputStream(textDocument.getText());
    const lexer = new DaphneDSLGrammarLexer(inputStream);
    const tokenStream = new CommonTokenStream(lexer);
    const parser = new DaphneDSLGrammarParser(tokenStream);

    const syntaxListener = new SyntaxErrorListener(textDocument);
    parser.removeErrorListeners();
    parser.addErrorListener(syntaxListener);

    const tree = parser.script();

    diagnostics.push(...syntaxListener.getDiagnostics());

    // 🔔 Semantic analysis correctly hooked:
    const semanticAnalyzer = new SemanticAnalyzer();
    const semanticDiagnostics = semanticAnalyzer.analyze(tree);
    diagnostics.push(...semanticDiagnostics);

  } catch (err: any) {
    diagnostics.push({
      severity: DiagnosticSeverity.Error,
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
  const start = performance.now();

  if (!document) {
    return { kind: DocumentDiagnosticReportKind.Full, items: [] };
  }

  const diagnostics = await validateTextDocument(document);
  const end = performance.now();

  connection.console.log(`🕒 Diagnostics for ${document.uri} took ${(end - start).toFixed(2)} ms`);

  return { kind: DocumentDiagnosticReportKind.Full, items: diagnostics };
});

connection.onCompletion((_params: TextDocumentPositionParams): CompletionItem[] => {
  const keywords = ['def', 'import', 'let', 'if', 'else', 'for', 'while', 'match', 'return', 'true', 'false', 'null'];
  return keywords.map((kw, index) => ({
    label: kw,
    kind: CompletionItemKind.Keyword,
    data: index
  }));
});

connection.onCompletionResolve((item: CompletionItem): CompletionItem => {
  item.detail = 'Daphne DSL keyword';
  item.documentation = `Keyword \`${item.label}\``;
  return item;
});

connection.onHover((params): Hover | null => {
  const document = documents.get(params.textDocument.uri);
  if (!document) {return null;}

  const position = params.position;
  const text = document.getText();
  const offset = document.offsetAt(position);

  const hoverWordMatch = /\b\w+\b/g;
  let word: string | null = null;
  let match: RegExpExecArray | null;

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

connection.onDefinition((params: DefinitionParams): Location[] => {
  const document = documents.get(params.textDocument.uri);
  if (!document) {return [];}

  try {
    const text = document.getText();
    const offset = document.offsetAt(params.position);

    const wordMatch = /\b\w+\b/g;
    let word: string | null = null;
    let match: RegExpExecArray | null;

    while ((match = wordMatch.exec(text))) {
      if (offset >= match.index && offset <= match.index + match[0].length) {
        word = match[0];
        break;
      }
    }

    if (!word) {return [];}

    const inputStream = new ANTLRInputStream(text);
    const lexer = new DaphneDSLGrammarLexer(inputStream);
    const tokenStream = new CommonTokenStream(lexer);
    const parser = new DaphneDSLGrammarParser(tokenStream);
    const tree = parser.script();  // ❗ can throw

    const analyzer = new SemanticAnalyzer();
    analyzer.analyze(tree);
    const symbols = analyzer.getSymbols();
connection.console.log(`🔎 Symbol Table: ${JSON.stringify(Array.from(symbols.entries()))}`);
connection.console.log(`🧠 Looking for word: ${word}`);


    const line = symbols.get(word);
    if (line === undefined) {return [];}

    return [{
      uri: params.textDocument.uri,
      range: {
        start: { line, character: 0 },
        end: { line, character: 100 }
      }
    }];
  } catch (err) {
    connection.console.error(`💥 Go-to Definition failed: ${err}`);
    return [];
  }
});





documents.listen(connection);
connection.listen();
