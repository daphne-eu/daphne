/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SRC_PARSER_CANCELINGERRORLISTENER_H
#define SRC_PARSER_CANCELINGERRORLISTENER_H

#include "antlr4-runtime.h"
#include <util/ErrorHandler.h>

class CancelingErrorListener : public antlr4::BaseErrorListener {
private:
    void syntaxError(antlr4::Recognizer *recognizer,
                     antlr4::Token *offendingSymbol,
                     size_t line,
                     size_t charPositionInLine,
                     const std::string &msg,
                     std::exception_ptr e) override {
        std::stringstream ss;
        ss << recognizer->getInputStream()->getSourceName() << ':' << line << ':' << charPositionInLine << ' '
           << msg << "\n";
        throw ErrorHandler::makeError("Antlr4 Parser", ss.str(), recognizer->getInputStream()->getSourceName(), line, charPositionInLine);
        // throw antlr4::ParseCancellationException(ss.str());
    }
};

#endif //SRC_PARSER_CANCELINGERRORLISTENER_H
