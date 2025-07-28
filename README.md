# DaphneDSL VS Code Extension (LSP Prototype)

This repository contains the working prototype of a VS Code Language Server Protocol (LSP) extension for the DaphneDSL language. It is designed as part of the Large Scale Data Engineering (LDE) project.

---

## Getting Started

To set up and run the DaphneDSL extension in VS Code:

### 1. **Clone the Repository**

```bash
git clone <this-repo-url>
cd lsp-sample
```

### 2. **Install Dependencies**

Run the following command in the root folder to install dependencies for both server and client:

```bash
npm install
```

### 3. **Build the Extension**

```bash
npm run compile
```

### 4. **Launch the Extension**

Press `F5` in VS Code (or click "Run Extension") to start a new Extension Development Host window.

### 5. **Open a Test File**

Navigate to the `benchmarks/` folder and open `example.daphne` to see all working features in action.

---

## Project Structure

```
lsp-sample/
├── client/              # VS Code Extension (Language Client)
├── server/              # LSP Server (Node + ANTLR Parser)
│   └── server/
│       ├── SemanticAnalyzer.ts  # Main logic for diagnostics
├── grammar/             # DaphneDSLGrammar.g4 (ANTLR)
├── benchmarks/          # Test Daphne files
│   └── example.daphne
```

---

## ✅ Implemented LSP Features

### 1. **Syntax Highlighting**

* Based on the Daphne grammar (ANTLR).
* Keywords like `def`, `let`, `return`, `as`, `true`, `false` are highlighted.

### 2. **Diagnostics**

* **Unused Variable Warning**: Detects declared but never used variables.
* **Unknown Keyword Error**: Flags misspelled or invalid keywords (e.g., `iff` instead of `if`).
* **Type Mismatch Warnings** *(partial support)*: If a return statement type mismatches the declared return type.

> See the Problems panel (View → Problems) for diagnostics when editing `.daphne` files.

### 3. **Autocomplete**

* Suggests known Daphne keywords like `def`, `let`, `import`, etc.
* Works automatically when typing.

### 4. **Hover (basic)**

* Basic hover functionality inherited from syntax tree.

---

## ⚠️ Not Included

* ❌ Go-to-definition
* ❌ Hover with semantic info
* ❌ Signature help or symbol navigation

These can be added in future versions if deeper compiler integration is implemented.

---

## Testing

To test the extension, use the provided file:

### `benchmarks/example.daphne`

```daphne
// ✅ Valid function with variables and return
def main() as int {
    let x = 5;
    let y = 10;
    let unusedVar = 42;         // ⚠️ Should trigger "declared but never used"
    let result = add(x, y);
    return result;
}

def add(a as int, b as int) as int {
    return a + b;
}

// ❌ Misspelled keyword
iff true {                      // ❌ Should trigger "unknown keyword"
    let z = 1;
}
```

---

## Tech Stack

* [VS Code Extension API](https://code.visualstudio.com/api)
* [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
* [ANTLR 4](https://www.antlr.org/)
* TypeScript (Node.js)

---

## 📅 Project Timeline

* ✅ Grammar Integration: Done
* ✅ Syntax Highlighting: Done
* ✅ Diagnostics & Autocomplete: Done
* ⏳ Hover (advanced) & Definition Lookup: Future Work

---
