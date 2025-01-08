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

#pragma once
#include <api/cli/DaphneUserConfig.h>
#include <nlohmannjson/json.hpp>
#include <runtime/local/vectorized/LoadPartitioning.h>
#include <runtime/local/kernels/SIMDOperatorsDAPHNE/VectorExtensions.h>
#include <string>

// must be in the same namespace as the enum SelfSchedulingScheme
NLOHMANN_JSON_SERIALIZE_ENUM(SelfSchedulingScheme, {{SelfSchedulingScheme::INVALID, nullptr},
                                                    {SelfSchedulingScheme::STATIC, "STATIC"},
                                                    {SelfSchedulingScheme::SS, "SS"},
                                                    {SelfSchedulingScheme::GSS, "GSS"},
                                                    {SelfSchedulingScheme::TSS, "TSS"},
                                                    {SelfSchedulingScheme::FAC2, "FAC2"},
                                                    {SelfSchedulingScheme::TFSS, "TFSS"},
                                                    {SelfSchedulingScheme::FISS, "FISS"},
                                                    {SelfSchedulingScheme::VISS, "VISS"},
                                                    {SelfSchedulingScheme::PLS, "PLS"},
                                                    {SelfSchedulingScheme::MSTATIC, "MSTATIC"},
                                                    {SelfSchedulingScheme::MFSC, "MFSC"},
                                                    {SelfSchedulingScheme::PSS, "PSS"}})

// must be in the same namespace as the enum VectorExtensions
NLOHMANN_JSON_SERIALIZE_ENUM(VectorExtensions, {{VectorExtensions::INVALID, nullptr},
                                                {VectorExtensions::SCALAR, "SCALAR"},
                                                {VectorExtensions::SSE, "SSE"},
                                                {VectorExtensions::AVX2, "AVX2"},
                                                {VectorExtensions::AVX512, "AVX512"}})

class ConfigParser {
  public:
    static bool fileExists(const std::string &filename);
    static void readUserConfig(const std::string &filename, DaphneUserConfig &config);

  private:
    static bool keyExists(const nlohmann::json &j, const std::string &key);
    static void checkAnyUnexpectedKeys(const nlohmann::basic_json<> &j, const std::string &filename);
};
