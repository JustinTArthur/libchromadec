/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_ERRORS_H
#define CHROMADEC_ERRORS_H

#include <chromadec/enum.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef CHD_ENUM(chd_status) {
    CHD_OK                         = 0,
    CHD_E_INVALID_ARG              = 1,
    CHD_E_FILE_NOT_FOUND           = 2,
    CHD_E_IO                       = 3,
    CHD_E_FORMAT_UNSUPPORTED       = 4,
    CHD_E_METADATA_MISSING         = 5,
    CHD_E_METADATA_CORRUPT         = 6,
    CHD_E_PRESET_UNKNOWN           = 7,
    CHD_E_DECODER_UNKNOWN          = 8,
    CHD_E_DECODER_INCOMPATIBLE     = 9,
    CHD_E_NN_MODEL_LOAD            = 10,
    CHD_E_NN_BACKEND_UNAVAILABLE   = 11,
    CHD_E_NN_INFERENCE             = 12,
    CHD_E_OUT_OF_RANGE             = 13,
    CHD_E_CANCELLED                = 14,
    CHD_E_INTERNAL                 = 15,
    CHD_E_OOM                      = 16,
    CHD_E_UNSUPPORTED              = 17
} chd_status_t;

/* Stable static name for a status, e.g. "CHD_E_IO". Never NULL. */
const char *chd_status_str(chd_status_t s);

/* Detail for the calling thread's most recent failure. Never NULL; empty when
 * nothing has failed on this thread. Read it only after a call returned
 * something other than CHD_OK: a failure the library handled internally can
 * leave a detail behind, so a non-empty string does not by itself mean the
 * call you just made failed. Valid until the next chd_* call on this thread. */
const char *chd_last_error(void);
void        chd_clear_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
