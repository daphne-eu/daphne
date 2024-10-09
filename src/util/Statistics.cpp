/*
 * Copyright 2024 The DAPHNE Consortium
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

#include "Statistics.h"

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <map>

Statistics &Statistics::instance() {
    static Statistics INSTANCE;
    return INSTANCE;
}

void Statistics::startKernelTimer(int kId) {
    auto startTime = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lg(m_times);
    startTimes[kId] = startTime;
}

void Statistics::stopKernelTimer(int kId) {
    auto const stopTime = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lg(m_times);
    std::chrono::duration<double> kernelTime = stopTime - startTimes[kId];
    kernelExecutionTimes.push_back({kId, kernelTime});
}

size_t getMaxKernelNameLength(KernelDispatchMapping &kdm) {
    auto maxLen = 0ul;
    for (auto const &[kId, kdmInfo] : kdm) {
        maxLen = kdmInfo.kernelName.length() > maxLen ? kdmInfo.kernelName.length() : maxLen;
    }
    return maxLen;
}

std::vector<OperatorStatistics> Statistics::processStatisticsPerOperator(KernelDispatchMapping &kdm) {
    std::vector<OperatorStatistics> stats;
    std::map<int, OperatorStatistics> statsByKid;

    std::sort(begin(kernelExecutionTimes), end(kernelExecutionTimes),
              [](auto const &t1, auto const &t2) { return std::get<0>(t1) > std::get<0>(t2); });

    for (auto const &[kId, time] : kernelExecutionTimes) {
        statsByKid[kId] += {kdm.getKernelDispatchInfo(kId), 1, time.count()};
    }

    std::transform(statsByKid.begin(), statsByKid.end(), std::back_inserter(stats),
                   [](const std::map<int, OperatorStatistics>::value_type &pair) { return pair.second; });

    std::sort(rbegin(stats), rend(stats));

    return stats;
}

void Statistics::dumpStatistics(KernelDispatchMapping &kdm) {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("DAPHNE operator execution runtime statistics.");
    auto maxLen = getMaxKernelNameLength(kdm);

    spdlog::info("{:<2}  {:<{}}  {:<9}{:<11}  {}  {}", "#", "Operator Name", maxLen, "Time(s)", "Count", "Avg(s)",
                 "File:Line:Column");
    std::vector<OperatorStatistics> opStats = processStatisticsPerOperator(kdm);

    auto i = 0;
    for (auto const &[kdmInfo, count, opTime] : opStats) {
        spdlog::info("{:<2}  {:<{}}  {:<9.2f}{:<13}{:<8.2f}{}", i++, kdmInfo.kernelName, maxLen, opTime, count,
                     opTime / count, fmt::format("{}:{}:{}", kdmInfo.fileName, kdmInfo.line, kdmInfo.column));
        if (i > Statistics::MAX_STATS_COUNT)
            break;
    }
}
