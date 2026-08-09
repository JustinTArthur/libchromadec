/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_LOG_H
#define CHROMADEC_LOG_H

#include <chromadec/enum.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Diagnostic severities. CHD_LOG_OFF is a threshold value only and is never
 * passed to a sink. */
typedef CHD_ENUM(chd_log_level) {
    CHD_LOG_DEBUG = 0,
    CHD_LOG_INFO  = 1,
    CHD_LOG_WARN  = 2,
    CHD_LOG_ERROR = 3,
    CHD_LOG_OFF   = 4
} chd_log_level_t;

/* Facts about a diagnostic beyond its severity. A bitmask rather than an
 * enumeration, since a combination of enumerators is not itself an enumerator.
 * Unrecognised bits may appear in future versions; test the ones you know. */
typedef unsigned int chd_log_flags_t;

/* The message is also the calling thread's chd_last_error() detail, so the
 * failure it describes is already on its way back as a non-CHD_OK
 * chd_status_t. A consumer that reports failures from the return path can drop
 * these rather than say the same thing twice; one whose only output is a log
 * pane should keep them. Diagnostics the library handles itself never carry
 * this flag, so dropping flagged messages loses nothing you do not learn from
 * the status you are about to be handed. */
#define CHD_LOG_F_RETURNED 0x1u

/* message is NUL-terminated UTF-8 with no trailing newline, valid only for the
 * duration of the call. */
typedef void (*chd_log_fn)(chd_log_level_t level, chd_log_flags_t flags,
                           const char *message, void *user_data);

/* Install the diagnostic sink, or pass NULL to uninstall. No sink is installed
 * by default, so the library writes nothing anywhere until a consumer asks for
 * it. The sink may be called from internal worker threads and must be
 * thread-safe; it must not call back into libchromadec. Once this returns, the
 * previous sink is guaranteed not to be running or to be entered again, so it
 * is safe to free its user_data.
 *
 * The sink is process-wide, not per-handle: in a host that loads several
 * consumers of this library into one process, the last installer wins.
 *
 * Safe to call before chd_init and from an atexit handler or a static
 * destructor; the library's log state is never torn down. */
void chd_set_log_callback(chd_log_fn fn_or_null, void *user_data);

/* Install a built-in sink that writes to stderr, one line per message. A
 * convenience for command-line consumers; equivalent to a chd_set_log_callback
 * sink that does the same. */
void chd_log_to_stderr(void);

/* Suppress diagnostics below min_level. Defaults to CHD_LOG_INFO, so debug
 * output is opt-in. CHD_LOG_OFF suppresses everything. A value outside the
 * enum is clamped into it. */
void            chd_set_log_level(chd_log_level_t min_level);
chd_log_level_t chd_get_log_level(void);

/* Whether a message at this level would reach a sink. Guards the cost of
 * building diagnostic data the library would then discard. CHD_LOG_OFF is a
 * threshold, not something a message carries, so it always answers 0. */
int chd_log_is_enabled(chd_log_level_t level);

/* Stable static name for a level, e.g. "WARN". */
const char *chd_log_level_str(chd_log_level_t level);

#ifdef __cplusplus
}
#endif
#endif