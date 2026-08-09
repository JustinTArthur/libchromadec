// SPDX-License-Identifier: GPL-3.0-or-later
#include "log.h"

#include "error_state.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace chd::log {

namespace {

// Both atomics are constant-initialized and trivially destructible, so they
// are alive for the whole life of the process regardless of where this
// translation unit falls in static construction and destruction order.
//
// The threshold is a plain atomic so a suppressed message, which is most of
// them once debug is off, never touches the mutex.
std::atomic<Level> g_min_level{CHD_LOG_INFO};
std::atomic<bool>  g_have_sink{false};

struct SinkState {
    std::shared_mutex mutex;
    chd_log_fn        fn = nullptr;           // guarded by mutex
    void             *user_data = nullptr;    // guarded by mutex
};

// Deliberately never destroyed. A consumer may log from an atexit handler or
// from a static destructor of its own, either of which can run after this
// translation unit's static objects would otherwise have been torn down;
// destroying a shared_mutex out from under a later emit is undefined. Leaking
// one small object makes logging safe at any point in the process's life
// rather than merely safe most of the time.
SinkState &sink() {
    static SinkState *state = new SinkState();
    return *state;
}

// Emitting takes the shared lock and reconfiguring takes the exclusive one, so
// a sink can never be torn down underneath a call already running. That is
// what lets a consumer free its user_data (or, from Rust, drop the boxed
// closure) as soon as chd_set_log_callback returns.
void dispatch(Level level, Flags flags, const char *message) {
    SinkState &s = sink();
    std::shared_lock<std::shared_mutex> lock(s.mutex);
    if (s.fn != nullptr) s.fn(level, flags, message, s.user_data);
}

// One insert, so lines from concurrent decode workers cannot interleave with
// each other. Individual stream inserts are race-free but not atomic as a
// group, and this sink is called from worker threads by design.
//
// Flags are ignored: this sink exists for a command-line consumer that wants
// everything on the terminal, and such a program is the one least likely to be
// reporting failures a second way.
void stderrSink(chd_log_level_t level, chd_log_flags_t, const char *message, void *) {
    std::string line = chd_log_level_str(level);
    line += ": ";
    line += message;
    line += '\n';
    std::cerr << line;
}

}  // namespace

bool isEnabled(Level level) {
    // CHD_LOG_OFF is a threshold value; no message ever carries it, so asking
    // whether one would be delivered is always answered no.
    if (level >= CHD_LOG_OFF) return false;
    return g_have_sink.load(std::memory_order_relaxed) &&
           level >= g_min_level.load(std::memory_order_relaxed);
}

void emit(Level level, Flags flags, const char *message) {
    if (!isEnabled(level)) return;
    dispatch(level, flags, message);
}

void setSink(chd_log_fn fn_or_null, void *user_data) {
    SinkState &s = sink();
    std::unique_lock<std::shared_mutex> lock(s.mutex);
    s.fn = fn_or_null;
    s.user_data = user_data;
    g_have_sink.store(fn_or_null != nullptr, std::memory_order_relaxed);
}

void installStderrSink() { setSink(stderrSink, nullptr); }

void setMinLevel(Level level) {
    // The entry point takes an enum but a caller can hand over any int32_t,
    // so clamp to a defined threshold rather than store a value that would
    // make every later comparison meaningless.
    if (level < CHD_LOG_DEBUG) {
        level = CHD_LOG_DEBUG;
    } else if (level > CHD_LOG_OFF) {
        level = CHD_LOG_OFF;
    }
    g_min_level.store(level, std::memory_order_relaxed);
}

Level minLevel() { return g_min_level.load(std::memory_order_relaxed); }

Stream::Stream(Level level, bool record_as_last_error)
    : level_(level),
      active_(record_as_last_error || isEnabled(level)),
      record_(record_as_last_error) {}

Stream::Stream(Stream &&other) noexcept
    : buf_(std::move(other.buf_)),
      level_(other.level_),
      sep_(other.sep_),
      first_(other.first_),
      active_(other.active_),
      record_(other.record_) {
    other.active_ = false;
}

Stream::~Stream() {
    if (!active_) return;
    const std::string message = buf_.str();
    if (record_) chd::detail::set_last_error(message);
    // emit() re-checks rather than trusting the constructor's answer: the sink
    // may have been uninstalled while the message was being built.
    emit(level_, record_ ? CHD_LOG_F_RETURNED : 0u, message.c_str());
}

Stream debug() { return Stream(CHD_LOG_DEBUG); }
Stream info()  { return Stream(CHD_LOG_INFO); }
Stream warn()  { return Stream(CHD_LOG_WARN); }
Stream error() { return Stream(CHD_LOG_ERROR); }
Stream fail()  { return Stream(CHD_LOG_ERROR, true); }

}  // namespace chd::log