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

#ifndef TEST_API_CLI_UTILS_H
#define TEST_API_CLI_UTILS_H

#include <api/cli/StatusCode.h>

#include <string>
#include <sstream>

/**
 * @brief Reads the entire contents of a plain text file into a string.
 * 
 * Not intended to be used with large files.
 * 
 * @param filePath The path to the file to be read.
 * @return A string containing the entire contents of the file.
 */
std::string readTextFile(const std::string & filePath);

/**
 * @brief Executes the specified program with the given arguments and captures
 * `stdout`, `stderr`, and the status code.
 * 
 * @param out The stream where to direct the program's standard output.
 * @param err The stream where to direct the program's standard error.
 * @param execPath The path to the executable.
 * @param args The arguments to pass. Despite the variadic template, each
 * element should be of type `char *`. The first one should be the name of the
 * program itself. The last one does *not* need to be a null pointer.
 * @return The status code returned by the process, or `-1` if it did not exit
 * normally.
 */
template<typename... Args>
int runProgram(std::stringstream & out, std::stringstream & err, const char * execPath, Args ... args);

/**
 * @brief Executes the given DaphneDSL script with the command line interface
 * of the DAPHNE Prototype and captures `stdout`, `stderr`, and the status code.
 * 
 * @param out The stream where to direct the program's standard output.
 * @param err The stream where to direct the program's standard error.
 * @param scriptPath The path to the DaphneDSL script file to execute.
 * @return The status code returned by the process, or `-1` if it did not exit
 * normally.
 */
int runDaphne(std::stringstream & out, std::stringstream & err, const char * scriptPath);

/**
 * @brief Checks whether executing the given DaphneDSL script with the command
 * line interface of the DAPHNE Prototype returns the given status code.
 * 
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * @param exp The expected status code.
 */
void checkDaphneStatusCode(const std::string & scriptFilePath, StatusCode exp);

void checkDaphneStatusCode(const std::string & dirPath, const std::string & name, unsigned idx, StatusCode exp);

/**
 * @brief Compares the standard output of executing the given DaphneDSL script
 * with the command line interface of the DAPHNE Prototype to a reference text
 * file.
 * 
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 * 
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * @param refFilePath The path to the plain text file containing the reference
 * output.
 */
void compareDaphneToRef(const std::string & scriptFilePath, const std::string & refFilePath);

void compareDaphneToRef(const std::string & dirPath, const std::string & name, unsigned idx);

#endif //TEST_API_CLI_UTILS_H