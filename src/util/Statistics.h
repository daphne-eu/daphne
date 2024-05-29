#pragma once

#include <util/KernelDispatchMapping.h>

#include <chrono>
#include <string>
#include <vector>

using namespace std::chrono_literals;

/**
 * @brief Structure that aggregates data per operation. Has a copy of the
 * according KDMInfo and keeps track of invocation cound and total operator
 * execution time.
 */
struct OperatorStatistics {
    KDMInfo kdmInfo;
    size_t count;
    double total;

    OperatorStatistics &operator+=(OperatorStatistics &&os) noexcept {
        OperatorStatistics rhs = os;
        return *this += rhs;
    }

    OperatorStatistics &operator+=(OperatorStatistics &os) noexcept {
        kdmInfo = os.kdmInfo;
        count += os.count;
        total += os.total;
        return *this;
    }

    OperatorStatistics operator+(OperatorStatistics &rhs) noexcept {
        return *this += rhs;
    }

    constexpr bool operator<(const OperatorStatistics &os) noexcept {
        return std::tie(total) < std::tie(os.total);
    }

    constexpr bool operator<=(const OperatorStatistics &os) noexcept {
        return std::tie(total) <= std::tie(os.total);
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const OperatorStatistics &opStats);
};

inline std::ostream &operator<<(std::ostream &os,
                                const OperatorStatistics &opStats) {
    return os << "Name: " << opStats.kdmInfo.kernelName << "\n"
              << "Total: " << opStats.total << "\n"
              << "Count: " << opStats.count << "\n";
}

/**
 * @brief The Statistics class provides an API to allow kernel calls to track
 * their exeuction time. Statistics is a singleton since it needs to track
 * across multiple DaphneContext instances.
 * KernelStats and times have to be guarded by a mutex, otherwise these are not
 * thread-safe.
 * Dumps aggregated statistics of the DAPHNE script to stdout and includes the
 * following information:
 * - operator name
 * - time spent in seconds
 * - invocation cound of the operator
 * - average time spent per invocation
 * - file, line and column position information in the source file
 * Entries are printed in descending order of total time spent for the operand.
 *
 * TODO: MAX_STATS_COUNT should be adjustable by the user.
 */
class Statistics {
  private:
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using KernelStats =
        std::vector<std::tuple<int, std::chrono::duration<double>>>;

    static constexpr int MAX_STATS_COUNT = 10;
    std::mutex m_times;
    KernelStats kernelExecutionTimes;
    std::unordered_map<int, Time> startTimes;

    std::vector<OperatorStatistics>
    processStatisticsPerOperator(KernelDispatchMapping &kdm);

  public:
    static Statistics &instance();
    void startKernelTimer(int kId);
    void stopKernelTimer(int kId);
    void dumpStatistics(KernelDispatchMapping &kdm);
};
