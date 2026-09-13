// SPDX-License-Identifier: GPL-3.0-or-later
//
// The reader field cache (reader/field_cache.h). The cache exists so a field
// wanted twice — once by its own frame, once by its neighbor in the 3D
// decoders — is read once, but it must never grow with the length of the
// capture: without a bound it ends up holding the whole file, which is tens of
// gigabytes for a feature-length tape.

#include "../../src/reader/field_cache.h"

#include <cstdint>
#include <iostream>

using chd::reader::Data;
using chd::reader::FieldCache;

namespace {

int failures = 0;

void check(bool ok, const char *what)
{
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        failures++;
    }
}

Data field(int32_t index, size_t samples = 8)
{
    return Data(samples, static_cast<uint16_t>(index));
}

}  // namespace

int main()
{
    // A hit returns what was stored; a miss is null.
    {
        FieldCache cache(4);
        cache.insert(7, field(7));
        const Data *hit = cache.find(7);
        check(hit != nullptr, "stored field is found");
        check(hit != nullptr && hit->front() == 7, "stored field holds its own data");
        check(cache.find(8) == nullptr, "absent field misses");
    }

    // Past the bound, the oldest entry goes and the rest survive intact.
    {
        FieldCache cache(4);
        for (int32_t i = 0; i < 6; i++) cache.insert(i, field(i));
        check(cache.find(0) == nullptr, "evicted oldest field");
        check(cache.find(1) == nullptr, "evicted second-oldest field");
        for (int32_t i = 2; i < 6; i++) {
            const Data *hit = cache.find(i);
            check(hit != nullptr && hit->front() == static_cast<uint16_t>(i),
                  "recycled entry carries the right field's data");
        }
    }

    // Eviction is by use, not by insertion order: a re-read field stays.
    {
        FieldCache cache(3);
        cache.insert(0, field(0));
        cache.insert(1, field(1));
        cache.insert(2, field(2));
        check(cache.find(0) != nullptr, "field 0 still cached before eviction");
        cache.insert(3, field(3));
        check(cache.find(0) != nullptr, "recently used field survives eviction");
        check(cache.find(1) == nullptr, "least recently used field is evicted");
    }

    // Re-inserting a cached field replaces its data rather than duplicating it.
    {
        FieldCache cache(2);
        cache.insert(5, field(5));
        cache.insert(5, field(9));
        const Data *hit = cache.find(5);
        check(hit != nullptr && hit->front() == 9, "re-insert replaces the entry");
        cache.insert(6, field(6));
        check(cache.find(5) != nullptr && cache.find(6) != nullptr,
              "re-insert did not consume a second slot");
    }

    // A long sequential pass — a whole capture decoded front to back — leaves
    // only the bound's worth behind, and clear() drops even that.
    {
        FieldCache cache(100);
        for (int32_t i = 0; i < 200000; i++) cache.insert(i, field(i));
        check(cache.find(199999) != nullptr, "last field of a long pass is cached");
        check(cache.find(199899) == nullptr, "a full bound back is already evicted");
        cache.clear();
        check(cache.find(199999) == nullptr, "clear empties the cache");
    }

    // A zero bound is a disabled cache, not an out-of-bounds eviction.
    {
        FieldCache cache(0);
        cache.insert(1, field(1));
        check(cache.find(1) == nullptr, "a zero bound caches nothing");
    }

    if (failures == 0) std::cout << "test_field_cache: OK\n";
    return failures == 0 ? 0 : 1;
}
