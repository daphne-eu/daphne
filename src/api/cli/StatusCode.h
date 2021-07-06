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

#ifndef SRC_API_CLI_STATUSCODE_H
#define SRC_API_CLI_STATUSCODE_H

/**
 * @brief Possible status codes returned by the command line interface.
 * 
 * Note that this is deliberately not an `enum class`, because we frequently
 * need to use it as an integer.
 */
enum StatusCode {
    SUCCESS = 0,
    PARSER_ERROR,
};

#endif //SRC_API_CLI_STATUSCODE_H