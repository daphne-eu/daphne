import * as vscode from 'vscode';

export async function runBenchmarks(client: any) {
  const testFiles = [
    'benchmarks/test_50.daphne',
    'benchmarks/test_500.daphne',
    'benchmarks/test_5000.daphne'
  ];

  for (const file of testFiles) {
    try {
      const fullPath = vscode.Uri.joinPath(
        vscode.workspace.workspaceFolders![0].uri,
        file
      );

      const document = await vscode.workspace.openTextDocument(fullPath);
      await vscode.window.showTextDocument(document, { preview: false });

      const position = new vscode.Position(0, 0);
      const params = {
        textDocument: { uri: document.uri.toString() },
        position
      };

      const start = performance.now();
      await client.sendRequest('textDocument/completion', params);
      const duration = performance.now() - start;

      console.log(`[Benchmark] ${file} autocomplete took ${duration.toFixed(2)}ms`);
    } catch (err) {
      console.error(`❌ Failed to benchmark ${file}:`, err);
    }
  }
}
