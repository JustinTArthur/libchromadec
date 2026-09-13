// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef CHD_READER_FIELD_CACHE_H
#define CHD_READER_FIELD_CACHE_H

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <unordered_map>
#include <utility>

#include "source.h"

namespace chd::reader {

// Bounded LRU of whole fields, keyed by the source's own 0-based field index.
//
// Sources are read front to back, and a field is wanted by the frame it
// belongs to and again by its neighbor in the 3D decoders, so a window of
// recent fields serves every hit an unbounded map would. The bound is what
// makes the reader's memory a function of the decoder's reach rather than of
// the capture's length: a field is roughly half a megabyte, so caching them
// all costs over 100 GB on an hour of NTSC read as a luma/chroma pair.
class FieldCache {
public:
    // Comfortably past the widest field window any decoder asks for, and past
    // what the file pipeline's batched workers hold in flight between them.
    static constexpr size_t kDefaultMaxFields = 100;

    explicit FieldCache(size_t maxFields = kDefaultMaxFields) : maxFields(maxFields) {}

    // Null when absent. A hit is promoted to most-recently-used, so the
    // pointer is only valid until the next call.
    const Data *find(int32_t fieldIndex)
    {
        auto it = index.find(fieldIndex);
        if (it == index.end()) return nullptr;
        entries.splice(entries.begin(), entries, it->second);
        return &entries.front().second;
    }

    void insert(int32_t fieldIndex, const Data &data)
    {
        if (maxFields == 0) return;

        auto it = index.find(fieldIndex);
        if (it != index.end()) {
            it->second->second = data;
            entries.splice(entries.begin(), entries, it->second);
            return;
        }
        if (entries.size() >= maxFields) {
            // Recycle the evicted entry rather than freeing it: every field is
            // the same length, so the buffer it already holds fits the new one
            // and the steady state does no allocation at all.
            auto node = std::prev(entries.end());
            index.erase(node->first);
            node->first = fieldIndex;
            node->second.assign(data.begin(), data.end());
            entries.splice(entries.begin(), entries, node);
        } else {
            entries.emplace_front(fieldIndex, data);
        }
        index.emplace(fieldIndex, entries.begin());
    }

    void clear()
    {
        entries.clear();
        index.clear();
    }

private:
    size_t maxFields;
    // Most-recently-used at the front.
    std::list<std::pair<int32_t, Data>> entries;
    std::unordered_map<int32_t, std::list<std::pair<int32_t, Data>>::iterator> index;
};

}  // namespace chd::reader

#endif  // CHD_READER_FIELD_CACHE_H
