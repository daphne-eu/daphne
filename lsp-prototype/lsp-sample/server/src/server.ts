/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */
import {
	createConnection,
	TextDocuments,
	Diagnostic,
	DiagnosticSeverity,
	ProposedFeatures,
	InitializeParams,
	DidChangeConfigurationNotification,
	CompletionItem,
	CompletionItemKind,
	TextDocumentPositionParams,
	TextDocumentSyncKind,
	InitializeResult,
	type DefinitionParams,
	type Location,
	Range,
	Position,
	DocumentDiagnosticReportKind,
	type DocumentDiagnosticReport,
	type Hover
} from 'vscode-languageserver/node';

import {
	TextDocument
} from 'vscode-languageserver-textdocument';

const connection = createConnection(ProposedFeatures.all);
connection.console.log("Daphne LSP server started!");

const documents = new TextDocuments(TextDocument);

let hasConfigurationCapability = false;
let hasWorkspaceFolderCapability = false;
let hasDiagnosticRelatedInformationCapability = false;

connection.onInitialize((params: InitializeParams) => {
	const capabilities = params.capabilities;

	hasConfigurationCapability = !!(
		capabilities.workspace && !!capabilities.workspace.configuration
	);
	hasWorkspaceFolderCapability = !!(
		capabilities.workspace && !!capabilities.workspace.workspaceFolders
	);
	hasDiagnosticRelatedInformationCapability = !!(
		capabilities.textDocument &&
		capabilities.textDocument.publishDiagnostics &&
		capabilities.textDocument.publishDiagnostics.relatedInformation
	);

	const result: InitializeResult = {
		capabilities: {
			textDocumentSync: TextDocumentSyncKind.Incremental,
			definitionProvider: true,
			hoverProvider: true,
			completionProvider: {
				resolveProvider: true
			},
			diagnosticProvider: {
				interFileDependencies: false,
				workspaceDiagnostics: false
			}
		}
	};
	if (hasWorkspaceFolderCapability) {
		result.capabilities.workspace = {
			workspaceFolders: {
				supported: true
			}
		};
	}
	return result;
});

connection.onInitialized(() => {
	if (hasConfigurationCapability) {
		connection.client.register(DidChangeConfigurationNotification.type, undefined);
	}
	if (hasWorkspaceFolderCapability) {
		connection.workspace.onDidChangeWorkspaceFolders(_event => {
			connection.console.log('Workspace folder change event received.');
		});
	}
});

function getWordAt(position: Position, text: string): string {
	const lines = text.split(/\r?\n/g);
	const line = lines[position.line] || '';
	const wordRegex = /\b\w+\b/g;

	let match: RegExpExecArray | null;
	while ((match = wordRegex.exec(line))) {
		const start = match.index;
		const end = start + match[0].length;
		if (position.character >= start && position.character <= end) {
			return match[0];
		}
	}
	return '';
}

connection.onDefinition((params: DefinitionParams): Location[] => {
  const start = performance.now();

  const uri = params.textDocument.uri;
  const word = getWordAt(params.position, documents.get(uri)?.getText() || '');

  let locations: Location[] = [];
  if (word === 'test') {
    locations = [{
      uri,
      range: {
        start: { line: 2, character: 7 },
        end: { line: 2, character: 11 }
      }
    }];
  }

  const end = performance.now();
  connection.console.log(`⏱️ Definition handled in ${Math.round(end - start)} ms`);

  return locations;
});


connection.onHover((params): Hover | null => {
  const start = performance.now();

  const document = documents.get(params.textDocument.uri);
  if (!document) {return null;}

  const position = params.position;
  const line = document.getText().split(/\r?\n/g)[position.line] || '';
  const wordRegex = /\b\w+\b/g;
  let match: RegExpExecArray | null;
  let word: string | null = null;

  while ((match = wordRegex.exec(line))) {
    const startIdx = match.index;
    const endIdx = startIdx + match[0].length;
    if (position.character >= startIdx && position.character <= endIdx) {
      word = match[0];
      break;
    }
  }

  if (!word) {return null;}

  const hoverTexts: Record<string, string> = {
    method: '🛠️ `method` defines a new function.',
    return: '↩️ `return` returns a value from a function.',
    test: '`test` is a placeholder function.'
  };

  const contents = hoverTexts[word];
  const end = performance.now();
  connection.console.log(`⏱️ Hover handled in ${Math.round(end - start)} ms`);

  if (!contents) {return null;}

  return {
    contents: {
      kind: 'markdown',
      value: contents
    }
  };
});

connection.onCompletion(
  (_params: TextDocumentPositionParams): CompletionItem[] => {
    const start = performance.now();

    const result: CompletionItem[] = [
      { label: 'method', kind: CompletionItemKind.Keyword, data: 1 },
      { label: 'return', kind: CompletionItemKind.Keyword, data: 2 },
      { label: 'if', kind: CompletionItemKind.Keyword, data: 3 },
      { label: 'true', kind: CompletionItemKind.Constant, data: 4 }
    ];

    const end = performance.now();
    connection.console.log(`⏱️ Completion handled in ${Math.round(end - start)} ms`);

    return result;
  }
);


connection.onCompletionResolve(
  (item: CompletionItem): CompletionItem => {
    if (item.data === 1) {
      item.detail = 'Keyword: method';
      item.documentation = '`method` declares a function.';
    } else if (item.data === 2) {
      item.detail = 'Keyword: return';
      item.documentation = '`return` returns a value.';
    } else if (item.data === 3) {
      item.detail = 'Keyword: if';
      item.documentation = '`if` starts a conditional block.';
    } else if (item.data === 4) {
      item.detail = 'Constant: true';
      item.documentation = 'Boolean value `true`.';
    }
    return item;
  }
);

async function validateTextDocument(textDocument: TextDocument): Promise<Diagnostic[]> {
  const text = textDocument.getText();
  const diagnostics: Diagnostic[] = [];

  // --- Rule 1: Warn about all-uppercase words ---
  const uppercasePattern = /\b[A-Z]{2,}\b/g;
  let match: RegExpExecArray | null;

  while ((match = uppercasePattern.exec(text))) {
    diagnostics.push({
      severity: DiagnosticSeverity.Warning,
      range: {
        start: textDocument.positionAt(match.index),
        end: textDocument.positionAt(match.index + match[0].length)
      },
      message: `⚠️ "${match[0]}" is all uppercase.`,
      source: 'daphne-lsp'
    });
  }

  // --- Rule 2: Error on mismatched curly braces ---
  const openBraces = (text.match(/{/g) || []).length;
  const closeBraces = (text.match(/}/g) || []).length;

  if (openBraces !== closeBraces) {
    diagnostics.push({
      severity: DiagnosticSeverity.Error,
      range: {
        start: textDocument.positionAt(0),
        end: textDocument.positionAt(text.length)
      },
      message: `❌ Mismatched curly braces: found {${openBraces}} and {${closeBraces}}.`,
      source: 'daphne-lsp'
    });
  }

  return diagnostics;
}

import { performance } from 'perf_hooks';

connection.languages.diagnostics.on(async (params) => {
  const document = documents.get(params.textDocument.uri);
  const start = performance.now();

  if (!document) {
    return {
      kind: DocumentDiagnosticReportKind.Full,
      items: []
    };
  }

  const diagnostics = await validateTextDocument(document);

  const end = performance.now();
  const duration = (end - start).toFixed(2);
  connection.console.log(`🕒 Diagnostics for ${document.uri} took ${duration} ms`);

  return {
    kind: DocumentDiagnosticReportKind.Full,
    items: diagnostics
  };
});


// Make the text document manager listen on the connection
documents.listen(connection);

// Listen on the connection
connection.listen();


