/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_PARSER_SCOPEDSYMBOLTABLE_H
#define SRC_PARSER_SCOPEDSYMBOLTABLE_H

#include <mlir/IR/Value.h>

#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cstddef>

/**
 * @brief A hierarchical symbol table offering a stack of nested scopes.
 * 
 * Each scope is a single-level symbol table. A symbol table maps a variable
 * name (symbol) to the SSA value currently denoted by that name. A symbol
 * table is used during the parsing of a DSL script. This particular kind of
 * symbol table is intended to be used *within a function*, where there can be
 * nested scopes due to block statements and control structures like
 * if-then-else and loops.
 */
class ScopedSymbolTable {
    
public:
    /**
     * @brief The type of single-level symbol table.
     */
    using SymbolTable = std::unordered_map<std::string, mlir::Value>;
    
private:
    /**
     * @brief A stack of single-level symbol tables representing nested scopes.
     */
    std::vector<SymbolTable> scopes;
    
    /**
     * @brief Determines whether some scope before the current scope has the
     * given symbol.
     * 
     * @param sym The symbol (variable name) to look for.
     * @return `true` if the symbol is found, `false` otherwise.
     */
    bool someParentHas(const std::string & sym) {
        for(int i = scopes.size() - 2; i >= 0; i--)
            if(scopes[i].count(sym))
                return true;
        return false;
    }
    
public:
    /**
     * @brief Creates a new `ScopedSymbolTable` initialized with a single empty
     * scope.
     */
    ScopedSymbolTable() {
        pushScope();
    }
    
    /**
     * @brief Returns the SSA value associated with the given symbol, or throws
     * an exception if the symbol is unknown.
     * 
     * Starting at the current scope, all hierarchy levels are searched until
     * the first occurrence of the symbol is found.
     * 
     * @param sym The symbol (variable name) to look for.
     * @return The associated SSA value.
     */
    mlir::Value get(const std::string & sym) {
        for(int i = scopes.size() - 1; i >= 0; i--) {
            auto it = scopes[i].find(sym);
            if(it != scopes[i].end())
                return it->second;
        }
        throw std::runtime_error("symbol not found: '" + sym + "'");
    }
    
    /**
     * @brief Like the other `get` method, but first tries to find the symbol
     * in the given single-level symbol table.
     * 
     * @param sym The symbol (variable name) to look for.
     * @param tab A single-level symbol table from outside of this
     * `ScopedSymbolTable`.
     * @return The associated SSA value.
     */
    mlir::Value get(const std::string & sym, const SymbolTable & tab) {
        auto it = tab.find(sym);
        if(it != tab.end())
            return it->second;
        return get(sym);
    }
    
    /**
     * @brief Associates the given SSA value with the given symbol.
     * 
     * The association is always created in the current scope. Any existing
     * mapping in that scope will be overwritten.
     * 
     * @param sym The symbol (variable name).
     * @param val The SSA value.
     */
    void put(std::string sym, mlir::Value val) {
        scopes.back()[sym] = val;
    }
    
    /**
     * @brief Puts all symbol-to-value mappings in the given single-level
     * symbol table into the current scope.
     * 
     * Existing mappings are overwritten in case of duplicate symbols.
     * 
     * @param tab The single-level symbol table to read from.
     */
    void put(SymbolTable tab) {
        for(auto it = tab.begin(); it != tab.end(); it++)
            put(it->first, it->second);
    }
    
    /**
     * @brief Creates a new scope in the hierarchy of nested scopes.
     * 
     * All subsequent calls to `get` and `put` will address the new scope.
     */
    void pushScope() {
        scopes.push_back(SymbolTable());
    }
    
    /**
     * @brief Removes the current scope from the hierarchy of nested scopes.
     * 
     * @return A single-level symbol table containing only those symbols that
     * (1) existed prior to the removed scope, and (2) were overwritten in the
     * removed scope.
     */
    SymbolTable popScope() {
        SymbolTable curScope = scopes.back();
        SymbolTable overwritten;
        for(auto it = curScope.begin(); it != curScope.end(); it++) {
            if(someParentHas(it->first))
                overwritten[it->first] = it->second;
        }
        scopes.pop_back();
        return overwritten;
    }
    
    /**
     * @brief Prints the contents of this `ScopedSymbolTable` to a stream.
     * 
     * @param os The stream to print to. Could be `std::cout`.
     */
    void dump(std::ostream & os) {
        for(size_t i = 0; i < scopes.size(); i++) {
            os << "scope #" << i << ':' << std::endl;
            for(auto it = scopes[i].begin(); it != scopes[i].end(); it++)
                os << '\t' << it->first << std::endl;
        }
        os << std::endl;
    }
    
    /**
     * @brief Determines the union of the symbols in the two given single-level
     * symbol tables.
     * 
     * @param lhs Some single-level symbol table.
     * @param rhs Some single-level symbol table.
     * @return The union of the symbol in the two input symbol tables.
     */
    static std::set<std::string> mergeSymbols(SymbolTable lhs, SymbolTable rhs) {
        std::set<std::string> res;
        
        for(auto it = lhs.begin(); it != lhs.end(); it++)
            res.insert(it->first);
        for(auto it = rhs.begin(); it != rhs.end(); it++)
            res.insert(it->first);
        
        return res;
    }
};

#endif //SRC_PARSER_SCOPEDSYMBOLTABLE_H