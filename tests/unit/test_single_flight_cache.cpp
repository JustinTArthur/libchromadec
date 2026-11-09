// SPDX-License-Identifier: GPL-3.0-or-later
//
// Contract of SingleFlightCache: results are computed once, failures are never
// stored, and threads waiting on a failure try for themselves.

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../src/common/single_flight_cache.h"

namespace {

#define REQUIRE(cond)                                                           \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using Cache = chd::SingleFlightCache<int, int>;

Cache::ValuePtr value(int v) { return std::make_shared<const int>(v); }

int testHitAndMiss() {
    Cache cache(2);
    int computed = 0;
    const auto compute = [&] { computed++; return value(7); };

    REQUIRE(*cache.getOrCompute(1, compute) == 7);
    REQUIRE(*cache.getOrCompute(1, compute) == 7);
    REQUIRE(computed == 1);
    REQUIRE(cache.stats().misses == 1);
    REQUIRE(cache.stats().hits == 1);
    return 0;
}

int testEvictsLeastRecentlyUsed() {
    Cache cache(2);
    int computed = 0;
    const auto compute = [&] { computed++; return value(computed); };

    cache.getOrCompute(1, compute);
    cache.getOrCompute(2, compute);
    cache.getOrCompute(1, compute);  // 2 is now the oldest
    const auto evictedHolder = cache.getOrCompute(2, compute);
    cache.getOrCompute(3, compute);  // evicts 1
    REQUIRE(computed == 3);
    REQUIRE(cache.stats().evictions == 1);

    cache.getOrCompute(2, compute);
    REQUIRE(computed == 3);
    cache.getOrCompute(1, compute);
    REQUIRE(computed == 4);
    REQUIRE(*evictedHolder == 2);
    return 0;
}

int testCapacityOnlyGrows() {
    Cache cache(1);
    int computed = 0;
    const auto compute = [&] { computed++; return value(computed); };

    cache.ensureCapacity(2);
    cache.ensureCapacity(1);
    cache.getOrCompute(1, compute);
    cache.getOrCompute(2, compute);
    cache.getOrCompute(1, compute);
    REQUIRE(computed == 2);
    return 0;
}

int testFailureNotStored() {
    Cache cache(2);
    int computed = 0;
    const auto fail = [&] { computed++; return Cache::ValuePtr(); };

    REQUIRE(cache.getOrCompute(1, fail) == nullptr);
    REQUIRE(cache.getOrCompute(1, fail) == nullptr);
    REQUIRE(computed == 2);
    return 0;
}

int testExceptionReleasesFlight() {
    Cache cache(2);
    bool threw = false;
    try {
        cache.getOrCompute(1, []() -> Cache::ValuePtr { throw std::runtime_error("boom"); });
    } catch (const std::runtime_error &) {
        threw = true;
    }
    REQUIRE(threw);
    REQUIRE(*cache.getOrCompute(1, [] { return value(3); }) == 3);
    return 0;
}

// Holds the leader inside compute() while other threads pile up on the key.
template <typename Finish>
int runContended(Cache &cache, int waiters, Finish finish, std::vector<Cache::ValuePtr> &results,
                 std::atomic<int> &computed) {
    std::promise<void> release;
    std::promise<void> leaderEntered;
    auto released = release.get_future().share();

    std::thread leader([&] {
        results[0] = cache.getOrCompute(1, [&] {
            computed++;
            leaderEntered.set_value();
            released.wait();
            return finish(true);
        });
    });
    leaderEntered.get_future().wait();

    std::vector<std::thread> others;
    for (int i = 1; i <= waiters; i++) {
        others.emplace_back([&, i] {
            results[i] = cache.getOrCompute(1, [&] {
                computed++;
                return finish(false);
            });
        });
    }
    // Long enough for the others to reach the wait. A late arrival sees the
    // same outcome, so this can't make the test fail.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    release.set_value();

    leader.join();
    for (auto &t : others) t.join();
    return 0;
}

int testWaitersShareResult() {
    Cache cache(2);
    std::atomic<int> computed{0};
    std::vector<Cache::ValuePtr> results(5);
    runContended(cache, 4, [](bool) { return value(9); }, results, computed);

    REQUIRE(computed == 1);
    for (const auto &r : results) REQUIRE(r != nullptr && *r == 9);
    REQUIRE(cache.stats().misses == 1);
    REQUIRE(cache.stats().hits == 4);
    return 0;
}

int testWaitersRetryAfterFailure() {
    Cache cache(2);
    std::atomic<int> computed{0};
    std::vector<Cache::ValuePtr> results(5);
    runContended(cache, 4, [](bool isLeader) { return isLeader ? Cache::ValuePtr() : value(5); }, results, computed);

    // The leader failed, one waiter then computed, and the rest shared that.
    REQUIRE(computed == 2);
    REQUIRE(results[0] == nullptr);
    for (int i = 1; i <= 4; i++) REQUIRE(results[i] != nullptr && *results[i] == 5);
    return 0;
}

int testDistinctKeysDontBlock() {
    Cache cache(4);
    std::promise<void> release;
    std::promise<void> leaderEntered;
    auto released = release.get_future().share();

    std::thread leader([&] {
        cache.getOrCompute(1, [&] {
            leaderEntered.set_value();
            released.wait();
            return value(1);
        });
    });
    leaderEntered.get_future().wait();

    // Finishes while key 1 is still in flight, or the test hangs.
    REQUIRE(*cache.getOrCompute(2, [] { return value(2); }) == 2);
    release.set_value();
    leader.join();
    return 0;
}

int testManyThreads() {
    Cache cache(8);
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 16; t++) {
        threads.emplace_back([&cache, &failures, t] {
            for (int i = 0; i < 2000; i++) {
                const int key = (i * (t + 1)) % 32;
                const auto result = cache.getOrCompute(key, [key] { return value(key * 3); });
                if (result == nullptr || *result != key * 3) failures++;
            }
        });
    }
    for (auto &t : threads) t.join();
    REQUIRE(failures == 0);
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= testHitAndMiss();
    rc |= testEvictsLeastRecentlyUsed();
    rc |= testCapacityOnlyGrows();
    rc |= testFailureNotStored();
    rc |= testExceptionReleasesFlight();
    rc |= testWaitersShareResult();
    rc |= testWaitersRetryAfterFailure();
    rc |= testDistinctKeysDontBlock();
    rc |= testManyThreads();
    if (rc == 0) std::cout << "All SingleFlightCache tests passed.\n";
    return rc;
}
