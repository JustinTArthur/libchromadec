// SPDX-License-Identifier: GPL-3.0-or-later
//
// Dropout options + last-frame stats accessor on the C ABI.
//
// The actual correction algorithm lives in src/dropout/dropout_corrector.cpp
// and is invoked by the pipeline. This only wires the per-decoder
// state path: callers set options on a chd_decoder_t handle once (or
// repeatedly between decode calls); the pipeline reads them when it
// instantiates a per-worker DropoutCorrector; the stats accessor returns
// the most recent frame's accumulated counts.
//
// chd_decode_frame and chd_decoder_commit themselves stay stubbed out
// until a follow-up; the dropout stats remain zero until those wire up.

#include <chromadec/dropout.h>

#include <cstring>

#include "../common/error_state.h"
#include "handles.h"

extern "C" {

chd_status_t chd_decoder_set_dropout(chd_decoder_t *d, const chd_dropout_opts_t *opts) {
    if (d == nullptr || opts == nullptr) {
        chd::detail::set_last_error("chd_decoder_set_dropout: null argument");
        return CHD_E_INVALID_ARG;
    }
    d->dropoutOpts = *opts;
    d->dropoutOptsSet = true;
    d->dropoutRevision.fetch_add(1, std::memory_order_release);
    return CHD_OK;
}

chd_status_t chd_decoder_get_last_dropout_stats(const chd_decoder_t *d,
                                                 chd_dropout_stats_t *out) {
    if (d == nullptr || out == nullptr) {
        chd::detail::set_last_error("chd_decoder_get_last_dropout_stats: null argument");
        return CHD_E_INVALID_ARG;
    }
    *out = d->lastDropoutStats;
    return CHD_OK;
}

}  // extern "C"
