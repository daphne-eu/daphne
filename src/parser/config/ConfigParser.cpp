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

#include <parser/config/ConfigParser.h>
#include <parser/config/JsonParams.h>

#include <iostream>
#include <fstream>

bool ConfigParser::fileExists(const std::string& filename) {
    // Open the given config file.
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("could not open file '" + filename + "' for reading user config");
    return true;
}

void ConfigParser::readUserConfig(const std::string& filename, DaphneUserConfig& config) {
    std::ifstream ifs(filename);
    nlohmann::basic_json jf = nlohmann::json::parse(ifs);

//try {
    checkAnyUnexpectedKeys(jf, filename);   // raise an error if the config JSON file contains any unexpected keys

    if (keyExists(jf, DaphneConfigJsonParams::USE_CUDA_))
        config.use_cuda = jf.at(DaphneConfigJsonParams::USE_CUDA_).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::USE_VECTORIZED_EXEC))
        config.use_vectorized_exec = jf.at(DaphneConfigJsonParams::USE_VECTORIZED_EXEC).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::USE_OBJ_REF_MGNT))
        config.use_obj_ref_mgnt = jf.at(DaphneConfigJsonParams::USE_OBJ_REF_MGNT).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::CUDA_FUSE_ANY))
        config.cuda_fuse_any = jf.at(DaphneConfigJsonParams::CUDA_FUSE_ANY).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::VECTORIZED_SINGLE_QUEUE))
        config.vectorized_single_queue = jf.at(DaphneConfigJsonParams::VECTORIZED_SINGLE_QUEUE).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::DEBUG_LLVM))
        config.debug_llvm = jf.at(DaphneConfigJsonParams::DEBUG_LLVM).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_KERNELS))
        config.explain_kernels = jf.at(DaphneConfigJsonParams::EXPLAIN_KERNELS).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_LLVM))
        config.explain_llvm = jf.at(DaphneConfigJsonParams::EXPLAIN_LLVM).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_PARSING))
        config.explain_parsing = jf.at(DaphneConfigJsonParams::EXPLAIN_PARSING).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_PROPERTY_INFERENCE))
        config.explain_property_inference = jf.at(DaphneConfigJsonParams::EXPLAIN_PROPERTY_INFERENCE).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_SQL))
        config.explain_sql = jf.at(DaphneConfigJsonParams::EXPLAIN_SQL).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_VECTORIZED))
        config.explain_vectorized = jf.at(DaphneConfigJsonParams::EXPLAIN_VECTORIZED).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_OBJ_REF_MGNT))
        config.explain_obj_ref_mgnt = jf.at(DaphneConfigJsonParams::EXPLAIN_OBJ_REF_MGNT).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::TASK_PARTITIONING_SCHEME)) {
        config.taskPartitioningScheme = jf.at(DaphneConfigJsonParams::TASK_PARTITIONING_SCHEME).get<SelfSchedulingScheme>();
        if (config.taskPartitioningScheme == SelfSchedulingScheme::INVALID) {
            throw std::invalid_argument("Invalid value for enum \"SelfSchedulingScheme\"" + config.taskPartitioningScheme);
        }
    }
    if (keyExists(jf, DaphneConfigJsonParams::NUMBER_OF_THREADS))
        config.numberOfThreads = jf.at(DaphneConfigJsonParams::NUMBER_OF_THREADS).get<int>();
    if (keyExists(jf, DaphneConfigJsonParams::MINIMUM_TASK_SIZE))
        config.minimumTaskSize = jf.at(DaphneConfigJsonParams::MINIMUM_TASK_SIZE).get<int>();
#ifdef USE_CUDA
    if (keyExists(jf, DaphneConfigJsonParams::CUDA_DEVICES))
        config.cuda_devices = jf.at(DaphneConfigJsonParams::CUDA_DEVICES).get<std::vector<int>>();
#endif
    if (keyExists(jf, DaphneConfigJsonParams::LIB_DIR))
        config.libdir = jf.at(DaphneConfigJsonParams::LIB_DIR).get<std::string>();
    if (keyExists(jf, DaphneConfigJsonParams::LIBRARY_PATHS))
        config.library_paths = jf.at(DaphneConfigJsonParams::LIBRARY_PATHS).get<std::vector<std::string>>();
//} catch (const nlohmann::detail::type_error& ex) {
//    std::cerr << ex.what() << std::endl;
//} catch (const nlohmann::detail::out_of_range& ex) {
//    std::cerr << ex.what() << std::endl;
//} catch (const std::invalid_argument& ex) {
//    std::cerr << ex.what() << std::endl;
//}
}

bool ConfigParser::keyExists(const nlohmann::json& j, const std::string& key) {
    return j.find(key) != j.end();
}

void ConfigParser::checkAnyUnexpectedKeys(const nlohmann::basic_json<>& j, const std::string& filename) {
    for (auto&[key, val]: j.items()) {
        bool flag = false;
        for (auto &jsonParam: DaphneConfigJsonParams::JSON_PARAMS) {
            if (key == jsonParam) {
                flag = true;
                break;
            }
        }
        if (!flag) throw std::invalid_argument("Unexpected key '" + key + "' in '" + filename + "' file");
    }
}