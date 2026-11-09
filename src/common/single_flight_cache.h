// SPDX-License-Identifier: GPL-3.0-or-later
//
// A bounded, thread-safe cache of immutable results that computes each key at
// most once at a time. Threads asking for a key another thread is computing
// wait for that result instead of repeating the work.

#ifndef CHD_COMMON_SINGLE_FLIGHT_CACHE_H
#define CHD_COMMON_SINGLE_FLIGHT_CACHE_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace chd {

// `Key` needs operator<. Results are published only once complete, and stay
// valid for holders of the returned pointer after eviction.
template <typename Key, typename Value>
class SingleFlightCache {
public:
    using ValuePtr = std::shared_ptr<const Value>;

    struct Stats {
        uint64_t hits = 0;       // answered without computing, including after a wait
        uint64_t misses = 0;     // computed through getOrCompute
        uint64_t evictions = 0;
        uint64_t bypasses = 0;   // computed outside the cache; see noteBypass
    };

    explicit SingleFlightCache(size_t capacity) : capacity_(capacity) {}

    SingleFlightCache(const SingleFlightCache &) = delete;
    SingleFlightCache &operator=(const SingleFlightCache &) = delete;

    // Raise the entry limit. Never lowers it.
    void ensureCapacity(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capacity > capacity_) capacity_ = capacity;
    }

    // Return the result for `key`, calling `compute` if nobody has it or is
    // computing it. `compute` returns a null pointer to report failure, or
    // throws. A failure is never stored, and each thread waiting on it then
    // tries for itself, so every caller sees what it would have seen alone.
    //
    // A caller must not hold a flight of its own while calling this, or two
    // threads could wait on each other.
    template <typename Compute>
    ValuePtr getOrCompute(const Key &key, Compute &&compute) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            const auto stored = index_.find(key);
            if (stored != index_.end()) {
                entries_.splice(entries_.begin(), entries_, stored->second);
                stats_.hits++;
                return stored->second->second;
            }

            const auto running = flights_.find(key);
            if (running == flights_.end()) break;

            const std::shared_ptr<Flight> flight = running->second;
            finished_.wait(lock, [&flight] { return flight->done; });
            if (flight->value != nullptr) {
                stats_.hits++;
                return flight->value;
            }
        }

        const auto flight = std::make_shared<Flight>();
        flights_.emplace(key, flight);
        stats_.misses++;
        lock.unlock();

        // Finishes the flight on every exit, so waiters are released if
        // `compute` throws.
        FlightGuard guard{*this, key, flight, nullptr};
        guard.value = compute();
        return guard.value;
    }

    // Record that a request was computed without going through the cache.
    void noteBypass() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.bypasses++;
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    using Entries = std::list<std::pair<Key, ValuePtr>>;

    struct Flight {
        bool done = false;
        ValuePtr value;
    };

    struct FlightGuard {
        SingleFlightCache &cache;
        const Key &key;
        std::shared_ptr<Flight> flight;
        ValuePtr value;

        ~FlightGuard() { cache.finishFlight(key, *flight, std::move(value)); }
    };

    void finishFlight(const Key &key, Flight &flight, ValuePtr value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            flight.value = value;
            flight.done = true;
            flights_.erase(key);
            if (value != nullptr && capacity_ > 0) store(key, std::move(value));
        }
        finished_.notify_all();
    }

    void store(const Key &key, ValuePtr value) {
        entries_.emplace_front(key, std::move(value));
        index_[key] = entries_.begin();
        while (entries_.size() > capacity_) {
            index_.erase(entries_.back().first);
            entries_.pop_back();
            stats_.evictions++;
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable finished_;
    size_t capacity_;
    Entries entries_;  // most recently used first
    std::map<Key, typename Entries::iterator> index_;
    std::map<Key, std::shared_ptr<Flight>> flights_;
    Stats stats_;
};

}  // namespace chd

#endif  // CHD_COMMON_SINGLE_FLIGHT_CACHE_H
