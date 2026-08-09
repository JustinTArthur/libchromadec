// SPDX-License-Identifier: GPL-3.0-or-later
//
// Diagnostic sink ABI. Verifies the default-silent contract, threshold
// filtering, the threshold-only nature of CHD_LOG_OFF, CHD_LOG_F_RETURNED
// marking the failures that also travel the return path, sink hand-off and
// uninstall, and that a sink installed from one thread receives messages
// emitted on another.

#include <chromadec/log.h>
#include <chromadec/video.h>

#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../src/common/log.h"

namespace {

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        return 1; \
    } \
} while (0)

struct Record {
    chd_log_level_t level;
    chd_log_flags_t flags;
    std::string     message;
};

struct Capture {
    std::mutex mutex;
    std::vector<Record> records;

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        records.clear();
    }

    std::size_t size() {
        std::lock_guard<std::mutex> lock(mutex);
        return records.size();
    }
};

void captureSink(chd_log_level_t level, chd_log_flags_t flags, const char *message,
                 void *user_data) {
    auto *capture = static_cast<Capture *>(user_data);
    std::lock_guard<std::mutex> lock(capture->mutex);
    capture->records.push_back(Record{level, flags, message});
}

int testDefaultIsSilent() {
    // Nothing is installed until a consumer asks, so nothing is enabled.
    REQUIRE(chd_log_is_enabled(CHD_LOG_ERROR) == 0);
    REQUIRE(chd_log_is_enabled(CHD_LOG_INFO) == 0);
    REQUIRE(chd_get_log_level() == CHD_LOG_INFO);

    // Emitting with no sink must be a no-op rather than a crash.
    chd::log::error() << "dropped on the floor";
    return 0;
}

int testLevelsAndFormatting() {
    Capture capture;
    chd_set_log_callback(captureSink, &capture);
    chd_set_log_level(CHD_LOG_INFO);

    REQUIRE(chd_log_is_enabled(CHD_LOG_DEBUG) == 0);
    REQUIRE(chd_log_is_enabled(CHD_LOG_INFO) == 1);
    REQUIRE(chd_log_is_enabled(CHD_LOG_ERROR) == 1);

    chd::log::debug() << "below threshold";
    chd::log::info() << "two" << "words";
    chd::log::warn().nospace() << "no" << "gap";
    chd::log::error() << "boom";

    REQUIRE(capture.size() == 3);
    REQUIRE(capture.records[0].level == CHD_LOG_INFO);
    REQUIRE(capture.records[0].message == "two words");
    REQUIRE(capture.records[1].level == CHD_LOG_WARN);
    REQUIRE(capture.records[1].message == "nogap");
    REQUIRE(capture.records[2].level == CHD_LOG_ERROR);
    REQUIRE(capture.records[2].message == "boom");

    // Raising the threshold drops everything below it; CHD_LOG_OFF drops all.
    capture.clear();
    chd_set_log_level(CHD_LOG_ERROR);
    chd::log::warn() << "suppressed";
    chd::log::error() << "kept";
    REQUIRE(capture.size() == 1);

    capture.clear();
    chd_set_log_level(CHD_LOG_OFF);
    REQUIRE(chd_log_is_enabled(CHD_LOG_ERROR) == 0);
    chd::log::error() << "suppressed";
    REQUIRE(capture.size() == 0);

    // Lowering it again reaches debug.
    chd_set_log_level(CHD_LOG_DEBUG);
    chd::log::debug() << "verbose";
    REQUIRE(capture.size() == 1);
    REQUIRE(capture.records[0].level == CHD_LOG_DEBUG);

    chd_set_log_callback(nullptr, nullptr);
    chd_set_log_level(CHD_LOG_INFO);
    return 0;
}

int testUninstall() {
    Capture capture;
    chd_set_log_callback(captureSink, &capture);
    chd::log::error() << "heard";
    REQUIRE(capture.size() == 1);

    // Once this returns the sink is unreachable, which is what makes it safe
    // for a caller to free the user_data it handed over.
    chd_set_log_callback(nullptr, nullptr);
    chd::log::error() << "not heard";
    REQUIRE(capture.size() == 1);
    REQUIRE(chd_log_is_enabled(CHD_LOG_ERROR) == 0);
    return 0;
}

int testThreadedEmit() {
    Capture capture;
    chd_set_log_callback(captureSink, &capture);
    chd_set_log_level(CHD_LOG_INFO);

    constexpr int kThreads = 4;
    constexpr int kPerThread = 50;
    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([t]() {
            for (int i = 0; i < kPerThread; ++i) {
                chd::log::info() << "worker" << t << "message" << i;
            }
        });
    }
    for (auto &worker : workers) worker.join();

    REQUIRE(capture.size() == kThreads * kPerThread);
    chd_set_log_callback(nullptr, nullptr);
    return 0;
}

// CHD_LOG_OFF is a threshold, so nothing is ever delivered at it and the
// is-enabled query says so at every threshold, sink or no sink.
int testOffIsNeverDeliverable() {
    Capture capture;
    chd_set_log_callback(captureSink, &capture);

    chd_set_log_level(CHD_LOG_DEBUG);
    REQUIRE(chd_log_is_enabled(CHD_LOG_DEBUG) == 1);
    REQUIRE(chd_log_is_enabled(CHD_LOG_OFF) == 0);

    chd_set_log_level(CHD_LOG_OFF);
    REQUIRE(chd_log_is_enabled(CHD_LOG_OFF) == 0);

    chd_set_log_callback(nullptr, nullptr);
    REQUIRE(chd_log_is_enabled(CHD_LOG_OFF) == 0);
    chd_set_log_level(CHD_LOG_INFO);
    return 0;
}

// CHD_LOG_F_RETURNED separates a failure the caller is also being handed from
// one that only ever reaches the sink. Both arrive at CHD_LOG_ERROR, so the
// flag is the only thing a consumer can dedupe on.
int testReturnedFlag() {
    Capture capture;
    chd_set_log_callback(captureSink, &capture);
    chd_set_log_level(CHD_LOG_INFO);

    chd::log::error() << "sink only";
    chd::log::fail() << "on the return path too";
    REQUIRE(capture.size() == 2);
    REQUIRE(capture.records[0].level == CHD_LOG_ERROR);
    REQUIRE((capture.records[0].flags & CHD_LOG_F_RETURNED) == 0);
    REQUIRE(capture.records[1].level == CHD_LOG_ERROR);
    REQUIRE((capture.records[1].flags & CHD_LOG_F_RETURNED) != 0);

    // Nothing below error level claims to be on the return path.
    capture.clear();
    chd::log::info() << "commentary";
    chd::log::warn() << "fallback taken";
    REQUIRE(capture.size() == 2);
    REQUIRE(capture.records[0].flags == 0);
    REQUIRE(capture.records[1].flags == 0);

    chd_set_log_callback(nullptr, nullptr);
    return 0;
}

int testLevelNames() {
    REQUIRE(std::string(chd_log_level_str(CHD_LOG_DEBUG)) == "DEBUG");
    REQUIRE(std::string(chd_log_level_str(CHD_LOG_INFO)) == "INFO");
    REQUIRE(std::string(chd_log_level_str(CHD_LOG_WARN)) == "WARN");
    REQUIRE(std::string(chd_log_level_str(CHD_LOG_ERROR)) == "ERROR");
    REQUIRE(std::string(chd_log_level_str(CHD_LOG_OFF)) == "OFF");
    // 99 names no enumerator but is still a valid value of the enum type
    // itself (fixed int32_t underlying type); it takes the fallback.
    REQUIRE(std::string(chd_log_level_str(static_cast<chd_log_level_t>(99))) == "UNKNOWN");
    return 0;
}

// Out-of-range thresholds clamp into the enumeration from both sides.
int testLevelClamp() {
    chd_set_log_level(static_cast<chd_log_level_t>(99));
    REQUIRE(chd_get_log_level() == CHD_LOG_OFF);
    chd_set_log_level(static_cast<chd_log_level_t>(-1));
    REQUIRE(chd_get_log_level() == CHD_LOG_DEBUG);
    chd_set_log_level(CHD_LOG_INFO);
    return 0;
}

// A failing open reports through the return path whether or not anyone is
// listening on the diagnostic channel.
int testFailureStillReportsWithoutSink() {
    chd_set_log_callback(nullptr, nullptr);
    chd_video_t *v = nullptr;
    const chd_status_t rc =
        chd_video_open_composite("/nonexistent/chromadec-log-test.tbc", nullptr, nullptr, &v);
    REQUIRE(rc != CHD_OK);
    REQUIRE(v == nullptr);
    REQUIRE(chd_last_error() != nullptr);
    REQUIRE(std::string(chd_last_error()).size() > 0);
    return 0;
}

}  // namespace

int main() {
    if (testDefaultIsSilent() != 0) return 1;
    if (testLevelsAndFormatting() != 0) return 1;
    if (testUninstall() != 0) return 1;
    if (testThreadedEmit() != 0) return 1;
    if (testOffIsNeverDeliverable() != 0) return 1;
    if (testReturnedFlag() != 0) return 1;
    if (testLevelNames() != 0) return 1;
    if (testLevelClamp() != 0) return 1;
    if (testFailureStillReportsWithoutSink() != 0) return 1;
    std::cout << "test_log_abi: all tests passed\n";
    return 0;
}