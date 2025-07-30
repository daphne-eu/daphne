import { DaphneDSLGrammarVisitor } from './DaphneDSLGrammarVisitor';
import { AbstractParseTreeVisitor } from 'antlr4ts/tree/AbstractParseTreeVisitor';
import { ParseTree } from 'antlr4ts/tree/ParseTree';
import { Diagnostic, DiagnosticSeverity } from 'vscode-languageserver';
import {
  FunctionRetTypesContext,
} from './DaphneDSLGrammarParser';

const VALID_KEYWORDS = new Set([
  'if', 'else', 'while', 'do', 'for', 'in', 'true', 'false',
  'as', 'def', 'return', 'import'
]);

export class SemanticAnalyzer extends AbstractParseTreeVisitor<void> implements DaphneDSLGrammarVisitor<void> {
  private declaredVars = new Map<string, { line: number; used: boolean }>();
  private diagnostics: Diagnostic[] = [];
  private symbolTable = new Map<string, number>();

  private currentFunctionReturnType: string | null = null;

  protected defaultResult() {
    return;
  }

  // Collect all diagnostics, especially for unused vars
  getDiagnostics(): Diagnostic[] {
    console.log("🔔 Collecting semantic diagnostics...");
    for (const [name, info] of this.declaredVars.entries()) {
      console.log(`  ➜ ${name} used=${info.used}`);
      if (!info.used) {
        this.diagnostics.push({
          severity: DiagnosticSeverity.Warning,
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

  visitScript(ctx: any) {
    console.log("📜 visitScript triggered");
    this.visitChildren(ctx);
  }

  visitStatement(ctx: any) {
    console.log("🧱 visitStatement triggered");
    this.visitChildren(ctx);
  }

  visitAssignStatement(ctx: any) {
    console.log("📌 visitAssignStatement triggered");
    const children = ctx.children;
    if (children && children.length >= 3) {
      const varToken = children[0] as any;
      const varName = typeof varToken.getText === 'function' ? varToken.getText() : 'unknown';
      const line = varToken.symbol?.line ?? 0;

      this.declaredVars.set(varName, { line: line - 1, used: false });
      this.symbolTable.set(varName, line - 1);
    }
    this.visitChildren(ctx);
  }

  visitVariableAccess(ctx: any) {
    const varName = ctx.text;
    const info = this.declaredVars.get(varName);
    if (info) {
      info.used = true;
    }
    this.visitChildren(ctx);
  }

  visitFunctionRetTypes(ctx: FunctionRetTypesContext) {
    const typeText = ctx.text;
    console.log(`📘 Return type extracted: ${typeText}`);
    this.currentFunctionReturnType = typeText;
    return this.visitChildren(ctx);
  }

  public analyze(tree: ParseTree): Diagnostic[] {
    this.visit(tree);
    return this.getDiagnostics();
  }

  public getSymbols(): Map<string, number> {
    return this.symbolTable;
  }

  private inferTypeFromText(text: string): string {
    if (/^\d+$/.test(text)) {return 'int';}
    if (/^".*"$/.test(text)) {return 'string';}
    if (/^\d+\.\d+$/.test(text)) {return 'float';}
    if (text === 'true' || text === 'false') {return 'bool';}
    return 'unknown';
  }
}
