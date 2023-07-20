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
#include <util/DaphneLogger.h>

#include <iostream>
#include <fstream>

int readLogLevel(const std::string& level) {
    std::string level_lowercase(level);
    std::transform(level.begin(), level.end(), level_lowercase.begin(), ::tolower);
    return static_cast<int>(spdlog::level::from_str(level_lowercase));
}

bool ConfigParser::fileExists(const std::string& filename) {
    // Open the given config file.
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("could not open file '" + filename + "' for reading user config");
    return true;
}

void ConfigParser::readUserConfig(const std::string& filename, DaphneUserConfig& config) {
    std::ifstream ifs(filename);
    auto jf = nlohmann::json::parse(ifs);

//try {
    checkAnyUnexpectedKeys(jf, filename);   // raise an error if the config JSON file contains any unexpected keys

    if (keyExists(jf, DaphneConfigJsonParams::USE_CUDA_))
        config.use_cuda = jf.at(DaphneConfigJsonParams::USE_CUDA_).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::USE_VECTORIZED_EXEC))
        config.use_vectorized_exec = jf.at(DaphneConfigJsonParams::USE_VECTORIZED_EXEC).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::USE_OBJ_REF_MGNT))
        config.use_obj_ref_mgnt = jf.at(DaphneConfigJsonParams::USE_OBJ_REF_MGNT).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::USE_IPA_CONST_PROPA))
        config.use_ipa_const_propa = jf.at(DaphneConfigJsonParams::USE_IPA_CONST_PROPA).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::USE_PHY_OP_SELECTION))
        config.use_phy_op_selection = jf.at(DaphneConfigJsonParams::USE_PHY_OP_SELECTION).get<bool>();
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
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_PARSING_SIMPLIFIED))
        config.explain_parsing_simplified = jf.at(DaphneConfigJsonParams::EXPLAIN_PARSING_SIMPLIFIED).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_PROPERTY_INFERENCE))
        config.explain_property_inference = jf.at(DaphneConfigJsonParams::EXPLAIN_PROPERTY_INFERENCE).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_SELECT_MATRIX_REPR))
        config.explain_select_matrix_repr = jf.at(DaphneConfigJsonParams::EXPLAIN_SELECT_MATRIX_REPR).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_SQL))
        config.explain_sql = jf.at(DaphneConfigJsonParams::EXPLAIN_SQL).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_PHY_OP_SELECTION))
        config.explain_phy_op_selection = jf.at(DaphneConfigJsonParams::EXPLAIN_PHY_OP_SELECTION).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_TYPE_ADAPTATION))
        config.explain_type_adaptation = jf.at(DaphneConfigJsonParams::EXPLAIN_TYPE_ADAPTATION).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_VECTORIZED))
        config.explain_vectorized = jf.at(DaphneConfigJsonParams::EXPLAIN_VECTORIZED).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::EXPLAIN_OBJ_REF_MGNT))
        config.explain_obj_ref_mgnt = jf.at(DaphneConfigJsonParams::EXPLAIN_OBJ_REF_MGNT).get<bool>();
    if (keyExists(jf, DaphneConfigJsonParams::TASK_PARTITIONING_SCHEME)) {
        config.taskPartitioningScheme = jf.at(DaphneConfigJsonParams::TASK_PARTITIONING_SCHEME).get<SelfSchedulingScheme>();
        if (config.taskPartitioningScheme == SelfSchedulingScheme::INVALID) {
            throw std::invalid_argument(std::string("Invalid value for enum \"SelfSchedulingScheme\"")
                    .append(std::to_string(static_cast<int>(config.taskPartitioningScheme))));
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
    if (keyExists(jf, DaphneConfigJsonParams::DAPHNEDSL_IMPORT_PATHS)) {
        config.daphnedsl_import_paths = jf.at(DaphneConfigJsonParams::DAPHNEDSL_IMPORT_PATHS).get<std::map<std::string,
                std::vector<std::string>>>();
    }
    if (keyExists(jf, DaphneConfigJsonParams::LOGGING)) {
        for (const auto&[key, val]: jf.at(DaphneConfigJsonParams::LOGGING).items()) {
            if(val.contains("log-level-limit")) {
                config.log_level_limit = static_cast<spdlog::level::level_enum>(readLogLevel(val.front()));
            }
            else if (val.contains("name")) {
                config.loggers.emplace_back(LogConfig({val.at("name"), val.at("filename"), readLogLevel(val.at("level")),
                        val.at("format")}));
            }
            else {
                spdlog::error("Not handling log config entry {}", key);
                for (const auto&[key2, val2]: val.items()) {
                    spdlog::error(key2);
                    spdlog::error(val2);
                }
            }

        }
    }
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
        if (!flag) {
            throw std::invalid_argument(std::string("Unexpected key '").append(key).append("' in '").append(filename)
                .append("' file"));
        }
    }
}