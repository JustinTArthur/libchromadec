/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * The public headers declare a C ABI, and this is the one translation unit in
 * the build that holds them to it: everything else the library compiles is C++
 * or Objective-C++, which accepts plenty of things a C compiler rejects.
 *
 * Every public header is included on its own first, so a header that stops
 * being self-contained fails here rather than in a consumer's build, and the
 * umbrella follows to catch a header missing from it. Taking the address of one
 * function per header puts each through the linker as well, which is what
 * confirms the symbol survived into the shared object's export list.
 */

#include <chromadec/decoder.h>
#include <chromadec/dropout.h>
#include <chromadec/errors.h>
#include <chromadec/frame.h>
#include <chromadec/log.h>
#include <chromadec/nn.h>
#include <chromadec/pipeline.h>
#include <chromadec/types.h>
#include <chromadec/version.h>
#include <chromadec/video.h>

#include <chromadec/chromadec.h>

#include <stdio.h>
#include <string.h>

/* Generic function-pointer type for the address-taking table below. Converting
 * between function-pointer types is fine as long as nothing is called through
 * the wrong one, and nothing here is called at all. */
typedef void (*chd_abi_fn)(void);

static chd_abi_fn const kExports[] = {
    (chd_abi_fn)chd_decoder_free,              /* decoder.h     */
    (chd_abi_fn)chd_dropout_spans_free,        /* dropout.h     */
    (chd_abi_fn)chd_status_str,                /* errors.h      */
    (chd_abi_fn)chd_frame_free,                /* frame.h       */
    (chd_abi_fn)chd_log_level_str,             /* log.h         */
    (chd_abi_fn)chd_nn_backend_is_available,   /* nn.h          */
    (chd_abi_fn)chd_cancel_create,             /* pipeline.h    */
    (chd_abi_fn)chd_version_string,            /* version.h     */
    (chd_abi_fn)chd_video_free,                /* video.h       */
};

/* types.h declares no functions, so the check it gets is that its opaque
 * handles and enums are usable as C types. */
static chd_video_t          *const kHandle   = NULL;
static const chd_plane_t           kPlane    = CHD_PLANE_Y;
static const chd_video_standard_t  kStandard = CHD_STD_NTSC;

static int failures = 0;

static void check(int ok, const char *what)
{
    if (!ok) {
        fprintf(stderr, "abi_smoke: %s\n", what);
        ++failures;
    }
}

int main(void)
{
    int major = -1, minor = -1, patch = -1;
    chd_cancel_t *cancel = NULL;

    for (size_t i = 0; i < sizeof kExports / sizeof kExports[0]; ++i)
        check(kExports[i] != NULL, "a public symbol resolved to NULL");

    check(kHandle == NULL && kPlane == CHD_PLANE_Y && kStandard == CHD_STD_NTSC,
          "types.h handles/enums did not survive C compilation");

    /* The headers have to agree with the library they were installed beside. */
    chd_version(&major, &minor, &patch);
    check(major == CHROMADEC_VERSION_MAJOR && minor == CHROMADEC_VERSION_MINOR &&
          patch == CHROMADEC_VERSION_PATCH,
          "chd_version() disagrees with the version macros");
    check(strcmp(chd_version_string(), CHROMADEC_VERSION_STRING) == 0,
          "chd_version_string() disagrees with CHROMADEC_VERSION_STRING");

    /* A call through each of the two shapes the ABI uses: a status return and
     * an out-parameter handle. */
    check(strcmp(chd_status_str(CHD_OK), "CHD_OK") == 0,
          "chd_status_str(CHD_OK) is not \"CHD_OK\"");

    check(chd_cancel_create(&cancel) == CHD_OK && cancel != NULL,
          "chd_cancel_create failed");
    if (cancel != NULL) {
        check(chd_cancel_is_requested(cancel) == 0, "a fresh cancel reads as requested");
        chd_cancel_request(cancel);
        check(chd_cancel_is_requested(cancel) != 0, "a requested cancel reads as clear");
        chd_cancel_free(cancel);
    }

    return failures == 0 ? 0 : 1;
}