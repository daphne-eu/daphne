import * as path from 'path';
import * as vscode from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
  const serverModule = context.asAbsolutePath(path.join('server', 'out', 'server.js'));

  const serverOptions: ServerOptions = {
    run: { module: serverModule, transport: TransportKind.ipc },
    debug: { module: serverModule, transport: TransportKind.ipc },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'daphne' }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/.clientrc')
    },
    outputChannelName: 'Daphne Language Client' // 🔥 This is critical
  };

  client = new LanguageClient(
    'daphneLanguageServer',
    'Daphne Language Client',
    serverOptions,
    clientOptions
  );

  client.start().then(() => {
    console.log('✅ Daphne LSP Client Started!');
  });
}

export function deactivate(): Thenable<void> | undefined {
  return client ? client.stop() : undefined;
}
