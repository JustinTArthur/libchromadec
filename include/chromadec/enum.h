/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_ENUM_H
#define CHROMADEC_ENUM_H

#include <stdint.h>

/* Declares a public enum. Where the language can express it (C++, C23), the
 * enum carries a fixed int32_t underlying type, so every int32_t value is a
 * valid value of the type and option validation can reject out-of-range input
 * in a well-defined way. Older C sees a plain enum with the same 4-byte
 * representation; the two views are ABI-identical. */
#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#define CHD_ENUM(name) enum name : int32_t
#else
#define CHD_ENUM(name) enum name
#endif

#endif
