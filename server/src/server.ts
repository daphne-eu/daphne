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
  TextDocumentPositionParams,
  DidChangeConfigurationNotification
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';
import { performance } from 'perf_hooks';

// 🧩 ANTLR integration:
// FIXED PATHS:
import { DaphneDSLGrammarLexer } from './server/DaphneDSLGrammarLexer';
import { DaphneDSLGrammarParser } from './server/DaphneDSLGrammarParser';

import { ANTLRInputStream, CommonTokenStream, RecognitionException } from 'antlr4ts';

const connection = createConnection(ProposedFeatures.all);
connection.console.log("✅ Daphne LSP server started!");

const documents = new TextDocuments(TextDocument);

connection.onInitialize((_params: InitializeParams) => {
  const result: InitializeResult = {
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
  return result;
});

async function validateTextDocument(textDocument: TextDocument): Promise<Diagnostic[]> {
  const text = textDocument.getText();
  const diagnostics: Diagnostic[] = [];

  try {
    const inputStream = new ANTLRInputStream(text);
    const lexer = new DaphneDSLGrammarLexer(inputStream);
    const tokenStream = new CommonTokenStream(lexer);
    const parser = new DaphneDSLGrammarParser(tokenStream);

    parser.script();
  } catch (err: any) {
    diagnostics.push({
      severity: DiagnosticSeverity.Error,
      range: {
        start: textDocument.positionAt(0),
        end: textDocument.positionAt(1)
      },
      message: `Syntax error detected.`,
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

connection.onDefinition((_params: DefinitionParams): Location[] => {
  return [];
});

documents.listen(connection);
connection.listen();
