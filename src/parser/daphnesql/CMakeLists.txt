# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

antlr4_generate(
        DaphneSQLGrammar
        ${CMAKE_CURRENT_SOURCE_DIR}/DaphneSQLGrammar.g4
        BOTH  # the specified g4 file contains both lexer and parser rules
        FALSE # don't generate listener interfaces
        TRUE  # generate visitor interfaces
)

include_directories(${ANTLR4_INCLUDE_DIR_DaphneSQLGrammar})

add_library(DaphneSQLParser STATIC
        DaphneSQLParser.cpp
        DaphneSQLVisitor.cpp
        ${ANTLR4_SRC_FILES_DaphneSQLGrammar}
)

target_link_libraries(DaphneSQLParser PRIVATE
        antlr4_static
)
