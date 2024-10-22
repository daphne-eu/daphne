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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/io/HDFS/HDFSUtils.h>

#if USE_HDFS
    #include <hdfs/hdfs.h>
#endif

#include <memory>
#include <vector>


// TODO: Separate implementation in a .cpp file?
class HDFSContext final : public IContext {
private:
    std::vector<std::string> workers{};
public:
#if USE_HDFS
    std::unique_ptr<hdfsFS> fs;
#endif
    HDFSContext(const DaphneUserConfig &cfg) {
#if USE_HDFS
        auto IpPort = HDFSUtils::parseIPAddress(cfg.hdfs_Address);
        fs = std::make_unique<hdfsFS>(hdfsConnectAsUser(std::get<0>(IpPort).c_str(), std::get<1>(IpPort), cfg.hdfs_username.c_str()));
#endif
    }
    ~HDFSContext() {
#if USE_HDFS
        hdfsDisconnect(*fs);
#endif
    };

    static std::unique_ptr<IContext> createHDFSContext(const DaphneUserConfig &cfg) {
        auto ctx = std::unique_ptr<HDFSContext>(new HDFSContext(cfg));
        return ctx;
    };

    void destroy() override {
        // Clean up
    };
#if USE_HDFS
    static HDFSContext* get(DaphneContext *ctx) { return dynamic_cast<HDFSContext*>(ctx->getHDFSContext()); };
    hdfsFS* getConnection() { return fs.get(); };
#endif
};