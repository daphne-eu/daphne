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

#ifndef SRC_PARSER_PARSER_H
#define SRC_PARSER_PARSER_H

#include <mlir/IR/Builders.h>

#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>


/**
 * @brief The super-class of all parsers producing DaphneIR.
 * 
 * All parsers generating a DaphneIR representation from a program or query
 * given in a particular domain-specific language (DSL) should inherit from
 * this class and implement `parseStream`.
 */
struct Parser {
    
    /**
     * @brief Generates a DaphneIR representation for the contents of the given
     * stream.
     * 
     * @param builder The builder to use for generating DaphneIR operations.
     * @param stream The stream to read from.
     * @param sourceName A name used for source location information.
     */
    virtual void parseStream(mlir::OpBuilder &builder, std::istream &stream, const std::string &sourceName) = 0;
    
    /**
     * @brief Generates a DaphneIR representation for the given DSL file.
     * 
     * @param builder The builder to use for generating DaphneIR operations.
     * @param filename The path to the file to read from.
     */
    void parseFile(mlir::OpBuilder & builder, const std::string & filename) {
        // Open the given DSL file.
        std::ifstream ifs(filename, std::ios::in);
        if (!ifs.good())
            throw std::runtime_error("could not open file '" + filename + "' for parsing");

        // Parse the file contents.
        parseStream(builder, ifs, filename);
    }

    /**
     * @brief Generates a DaphneIR representation for the given DSL string.
     * 
     * @param builder The builder to use for generating DaphneIR operations.
     * @param str The string to read from.
     * @param sourceName Optional name for the source used in MLIR Locations (defaults to "DSL String")
     */
    void parseStr(mlir::OpBuilder & builder, const std::string & str, const std::string & sourceName = "DSL String") {
        // Parse the file contents.
        std::istringstream s(str);
        
        // Parse the string contents.
        parseStream(builder, s, sourceName);
    }
};

#endif //SRC_PARSER_PARSER_H
