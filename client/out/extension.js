"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const path = require("path");
const vscode = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
function activate(context) {
    const serverModule = context.asAbsolutePath(path.join('server', 'out', 'server.js'));
    const serverOptions = {
        run: { module: serverModule, transport: node_1.TransportKind.ipc },
        debug: { module: serverModule, transport: node_1.TransportKind.ipc },
    };
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'daphne' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/.clientrc')
        },
        outputChannelName: 'Daphne Language Client' // 🔥 This is critical
    };
    client = new node_1.LanguageClient('daphneLanguageServer', 'Daphne Language Client', serverOptions, clientOptions);
    client.start().then(() => {
        console.log('✅ Daphne LSP Client Started!');
    });
}
function deactivate() {
    return client ? client.stop() : undefined;
}
//# sourceMappingURL=extension.js.map