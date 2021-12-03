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
#include <runtime/distributed/worker/WorkerImpl.h>

#include <catch.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <grpcpp/server.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <filesystem>

// TODO In all (run/check/compare)Daphne... functions, we might not need to
// explicitly specify the scriptPath as a parameter, since it could be subsumed
// by the parameter pack. Check if this is possible in all cases and simplify.

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
int runProgram(std::stringstream & out, std::stringstream & err, const char * execPath, Args ... args) {
    int linkOut[2]; // pipe ends for stdout
    int linkErr[2]; // pipe ends for stderr
    char buf[1024]; // internal buffer for reading from the pipes
    
    // Try to create the pipes.
    if(pipe(linkOut) == -1)
        throw std::runtime_error("could not create pipe");
    if(pipe(linkErr) == -1)
        throw std::runtime_error("could not create pipe");
    
    // Try to create the child process.
    pid_t p = fork();
    
    if(p == -1)
        throw std::runtime_error("could not create child process");
    else if(p) { // parent
        // Close write end of pipes.
        close(linkOut[1]);
        close(linkErr[1]);
        
        // Read data from stdout and stderr of the child from the pipes.
        ssize_t numBytes;
        while(numBytes = read(linkOut[0], buf, sizeof(buf)))
            out.write(buf, numBytes);
        while(numBytes = read(linkErr[0], buf, sizeof(buf)))
            err.write(buf, numBytes);
        
        // Wait for child's termination.
        int status;
        waitpid(p, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    else { // child
        // Redirect stdout and stderr to the pipe.
        dup2(linkOut[1], STDOUT_FILENO);
        dup2(linkErr[1], STDERR_FILENO);
        close(linkOut[0]);
        close(linkOut[1]);
        close(linkErr[0]);
        close(linkErr[1]);
        
        // Execute other program.
        execl(execPath, args..., static_cast<char *>(nullptr));
        
        // execl does not return, unless it failed.
        throw std::runtime_error("could not execute the program");
    }
}

/**
 * @brief Executes the specified program with the given arguments in the background.
 *  
 * @param out The file descriptor where to redirect standard output.
 * @param err The file descriptor where to redirect standard error.
 * @param execPath The path to the executable.
 * @param args The arguments to pass. Despite the variadic template, each
 * element should be of type `char *`. The first one should be the name of the
 * program itself. The last one does *not* need to be a null pointer.
 * @return The process id of the child process.
 * 
 */
template<typename... Args>
pid_t runProgramInBackground(int &out, int &err, const char * execPath, Args ... args) {        
    
    // Try to create the child process.
    pid_t p = fork();
    
    if(p == -1)
        throw std::runtime_error("could not create child process");
    else if(p) { // parent        
        // Return pid
        return p;
    }
    else { // child
        // Redirect stdout and stderr to the pipe.
        dup2(out, STDOUT_FILENO);
        dup2(err, STDERR_FILENO);
        // Execute other program.
        execl(execPath, args..., static_cast<char *>(nullptr));
        
        // execl does not return, unless it failed.
        throw std::runtime_error("could not execute the program");
    }
}


/**
 * @brief Executes the given DaphneDSL script with the command line interface
 * of the DAPHNE Prototype and captures `stdout`, `stderr`, and the status code.
 * 
 * @param out The stream where to direct the program's standard output.
 * @param err The stream where to direct the program's standard error.
 * @param scriptPath The path to the DaphneDSL script file to execute.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 * @return The status code returned by the process, or `-1` if it did not exit
 * normally.
 */
template<typename... Args>
int runDaphne(std::stringstream & out, std::stringstream & err, const char * scriptPath, Args ... args) {
    return runProgram(out, err, "build/bin/daphnec", "daphnec", scriptPath, args...);
}

/**
 * @brief Checks whether executing the given DaphneDSL script with the command
 * line interface of the DAPHNE Prototype returns the given status code.
 * 
 * @param exp The expected status code.
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 */
template<typename... Args>
void checkDaphneStatusCode(StatusCode exp, const std::string & scriptFilePath, Args ... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, scriptFilePath.c_str(), args...);

    REQUIRE(status == exp);
}

template<typename... Args>
void checkDaphneStatusCodeSimple(StatusCode exp, const std::string & dirPath, const std::string & name, unsigned idx, Args ... args) {
    checkDaphneStatusCode(exp, dirPath + name + '_' + std::to_string(idx) + ".daphne", args...);
}

/**
 * @brief Compares the standard output of executing the given DaphneDSL script
 * with the command line interface of the DAPHNE Prototype to a reference text.
 * 
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 * 
 * @param exp The expected output on stdout.
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * output.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 */
template<typename... Args>
void compareDaphneToStr(const std::string & exp, const std::string & scriptFilePath, Args ... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, scriptFilePath.c_str(), args...);

    REQUIRE(status == StatusCode::SUCCESS);
    CHECK(out.str() == exp);
    CHECK(err.str() == "");
}

/**
 * @brief Compares the standard output of executing the given DaphneDSL script
 * with the command line interface of the DAPHNE Prototype to a reference text
 * file.
 * 
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 * 
 * @param refFilePath The path to the plain text file containing the reference
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * output.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 */
template<typename... Args>
void compareDaphneToRef(const std::string & refFilePath, const std::string & scriptFilePath, Args ... args) {
    return compareDaphneToStr(readTextFile(refFilePath.c_str()), scriptFilePath, args...);
}

template<typename... Args>
void compareDaphneToRefSimple(const std::string & dirPath, const std::string & name, unsigned idx, Args ... args) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    compareDaphneToRef(filePath + ".txt", filePath + ".daphne", args...);
}

/**
 * @brief Compares the standard output of executing a given DaphneDSL script
 * with the command line interface of the DAPHNE Prototype, to a (simpler) DaphneDSL
 * script defining the expected behaviour.
 *
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 *
 * @param expScriptFilePath The path to the DaphneDSL script with expected behaviour.
 * @param actScriptFilePath The path to the DaphneDSL script to check with actual behaviour.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 */
template<typename... Args>
void compareDaphneToSelfRef(const std::string &expScriptFilePath, const std::string &actScriptFilePath, Args ... args) {
    std::stringstream expOut;
    std::stringstream expErr;
    int expStatus = runDaphne(expOut, expErr, expScriptFilePath.c_str(), args...);
    std::stringstream actOut;
    std::stringstream actErr;
    int actStatus = runDaphne(actOut, actErr, actScriptFilePath.c_str(), args...);

    REQUIRE(expStatus == actStatus);
    CHECK(expOut.str() == actOut.str());
    CHECK(expErr.str() == actErr.str());
}

template<typename... Args>
void compareDaphneToSelfRefSimple(const std::string & dirPath, const std::string & name, unsigned idx, Args ... args) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    compareDaphneToSelfRef(filePath + ".ref.daphne", filePath + ".daphne", args...);
}

/**
 * @brief Compares the standard output of executing a given DaphneDSL script
 * with a reference script or text file, based on which file is found.
 */
template<typename... Args>
void compareDaphneToSomeRefSimple(const std::string & dirPath, const std::string & name, unsigned idx, Args ... args) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    if (std::filesystem::exists(filePath + ".ref.daphne")) {
        compareDaphneToSelfRef(filePath + ".ref.daphne", filePath + ".daphne", args...);
    }
    else if (std::filesystem::exists(filePath + ".txt")) {
        compareDaphneToRef(filePath + ".txt", filePath + ".daphne", args...);
    }
    else {
        throw std::runtime_error("Could not find any ref for file `" + filePath + ".daphne`");
    }
}

/**
 * @brief Starts a distributed worker locally.
 *
 * @param addr The address (usually `0.0.0.0:<port>`) the worker should run on
 * @return The server (has to be kept around for the whole time the worker is in use)
 */
[[nodiscard]] std::unique_ptr<grpc::Server> startDistributedWorker(const char *addr, WorkerImpl *workerImpl);

#endif //TEST_API_CLI_UTILS_H