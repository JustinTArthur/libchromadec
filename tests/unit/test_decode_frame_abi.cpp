// SPDX-License-Identifier: GPL-3.0-or-later
//
// End-to-end test: drives the public C ABI through a full decode
// loop. Reuses the synthetic-TBC fixture pattern from
// test_encode_orc_color_bars (black frames + a minimal sqlite sidecar), and
// then exercises:
//   - chd_decoder_set_option_* + chd_decoder_has_option (validity by kind)
//   - chd_decoder_commit on a CHD_DEC_MONO decoder
//   - chd_decode_frame returning a YUV444P16 chd_frame_t
//   - chd_frame_get_info + chd_frame_get_plane (Y plane)
//   - chd_decode_frame out-of-range and not-committed error paths
//   - chd_decode_frames_async + chd_cancel_t cancel-before-dispatch
//
// Hardening additions:
//   - chd_video_add_extra_source_composite against a CVBS primary (previously
//     rejected with "primary source has no TBC metadata")
//   - Multi-source dropout path through chd_decode_frame with both primary
//     and extra TBC sources, verifying chd_decoder_get_last_dropout_stats
//
// Black input means the Y plane should sit at the YUV444P16 black point
// (Y_ZERO = 16 * 256 = 4096) and the Cb/Cr planes at C_ZERO = 128 * 256
// = 32768. We assert that to catch any future regression in the
// SourceField → Decoder → OutputWriter → chd_frame plumbing.

#include <chromadec/chromadec.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <sqlite3.h>

namespace fs = std::filesystem;

#define REQUIRE(cond)                                                            \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond \
                      << " (last_error=" << chd_last_error() << ")\n";           \
            return 1;                                                            \
        }                                                                        \
    } while (0)

namespace {

const char *kTbcSchema = R"(
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS capture (
    capture_id INTEGER PRIMARY KEY,
    system TEXT NOT NULL,
    decoder TEXT NOT NULL,
    git_branch TEXT,
    git_commit TEXT,
    video_sample_rate REAL,
    active_video_start INTEGER,
    active_video_end INTEGER,
    field_width INTEGER,
    field_height INTEGER,
    number_of_sequential_fields INTEGER,
    colour_burst_start INTEGER,
    colour_burst_end INTEGER,
    is_mapped INTEGER,
    is_subcarrier_locked INTEGER,
    is_widescreen INTEGER,
    white_16b_ire INTEGER,
    black_16b_ire INTEGER,
    blanking_16b_ire INTEGER,
    capture_notes TEXT
);

CREATE TABLE IF NOT EXISTS field_record (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    audio_samples INTEGER,
    decode_faults INTEGER,
    disk_loc REAL,
    efm_t_values INTEGER,
    field_phase_id INTEGER,
    file_loc INTEGER,
    is_first_field INTEGER,
    median_burst_ire REAL,
    pad INTEGER,
    sync_conf INTEGER,
    ntsc_is_fm_code_data_valid INTEGER,
    ntsc_fm_code_data INTEGER,
    ntsc_field_flag INTEGER,
    ntsc_is_video_id_data_valid INTEGER,
    ntsc_video_id_data INTEGER,
    ntsc_white_flag INTEGER
);

CREATE TABLE IF NOT EXISTS drop_outs (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    startx INTEGER NOT NULL,
    endx INTEGER NOT NULL,
    field_line INTEGER NOT NULL
);
)";

// One drop_outs row to inject into the synthetic sidecar. field_id is
// 0-based to match the sqlite-stored field index from the writeSidecar
// loop. (capture_id == 1 implicitly.)
struct DropoutRow {
    int32_t fieldId;
    int32_t startx;
    int32_t endx;
    int32_t fieldLine;
};

bool writeTbcSidecar(const std::string &path, int32_t numFields,
                     const std::vector<DropoutRow> &dropouts = {},
                     int32_t subcarrierLocked = 1) {
    if (fs::exists(path)) fs::remove(path);
    sqlite3 *db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) return false;
    if (sqlite3_exec(db, kTbcSchema, nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return false;
    }
    const char *capSql =
        "INSERT INTO capture (capture_id, system, decoder, video_sample_rate, "
        "  active_video_start, active_video_end, field_width, field_height, "
        "  number_of_sequential_fields, colour_burst_start, colour_burst_end, "
        "  is_mapped, is_subcarrier_locked, is_widescreen, "
        "  white_16b_ire, black_16b_ire, blanking_16b_ire) "
        "VALUES (1, 'NTSC', 'ld-decode', 14318181.818, 147, 905, 910, 263, ?, "
        "        92, 119, 1, ?, 0, 51200, 17920, 16384);";
    sqlite3_stmt *cap = nullptr;
    sqlite3_prepare_v2(db, capSql, -1, &cap, nullptr);
    sqlite3_bind_int(cap, 1, numFields);
    sqlite3_bind_int(cap, 2, subcarrierLocked);
    sqlite3_step(cap);
    sqlite3_finalize(cap);

    const char *fieldSql =
        "INSERT INTO field_record (capture_id, field_id, is_first_field, "
        "  sync_conf, median_burst_ire, ntsc_is_fm_code_data_valid, "
        "  ntsc_fm_code_data, ntsc_field_flag, ntsc_is_video_id_data_valid, "
        "  ntsc_video_id_data, ntsc_white_flag) "
        "VALUES (1, ?, ?, 100, 25.0, 0, 0, 0, 0, 0, 0);";
    sqlite3_stmt *fs = nullptr;
    sqlite3_prepare_v2(db, fieldSql, -1, &fs, nullptr);
    for (int32_t i = 0; i < numFields; i++) {
        sqlite3_reset(fs);
        sqlite3_bind_int(fs, 1, i);
        sqlite3_bind_int(fs, 2, (i % 2 == 0) ? 1 : 0);
        if (sqlite3_step(fs) != SQLITE_DONE) {
            sqlite3_finalize(fs);
            sqlite3_close(db);
            return false;
        }
    }
    sqlite3_finalize(fs);

    if (!dropouts.empty()) {
        const char *doSql =
            "INSERT INTO drop_outs (capture_id, field_id, startx, endx, field_line) "
            "VALUES (1, ?, ?, ?, ?);";
        sqlite3_stmt *doStmt = nullptr;
        sqlite3_prepare_v2(db, doSql, -1, &doStmt, nullptr);
        for (const auto &dr : dropouts) {
            sqlite3_reset(doStmt);
            sqlite3_bind_int(doStmt, 1, dr.fieldId);
            sqlite3_bind_int(doStmt, 2, dr.startx);
            sqlite3_bind_int(doStmt, 3, dr.endx);
            sqlite3_bind_int(doStmt, 4, dr.fieldLine);
            if (sqlite3_step(doStmt) != SQLITE_DONE) {
                sqlite3_finalize(doStmt);
                sqlite3_close(db);
                return false;
            }
        }
        sqlite3_finalize(doStmt);
    }
    sqlite3_close(db);
    return true;
}

bool writeUniformTbc(const std::string &path, int32_t w, int32_t h, int32_t n, uint16_t value) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const std::vector<uint16_t> buf(static_cast<size_t>(w) * h, value);
    for (int32_t i = 0; i < n; i++) {
        f.write(reinterpret_cast<const char *>(buf.data()), buf.size() * 2);
    }
    return true;
}

bool writeBlackTbc(const std::string &path, int32_t w, int32_t h, int32_t n) {
    return writeUniformTbc(path, w, h, n, 17920);  // blackIre from the sidecar
}

// Minimal CVBS .meta sidecar — one capture row in the cvbs_file table.
// Mirrors the helper in test_cvbs_reader.cpp.
bool writeCvbsMeta(const std::string &path, const std::string &preset,
                   const std::string &encoding, const std::string &state,
                   const std::string &signalType) {
    if (fs::exists(path)) fs::remove(path);
    sqlite3 *db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) return false;
    const char *schema = R"(
        PRAGMA user_version = 7;
        CREATE TABLE cvbs_file (
            cvbs_file_id INTEGER PRIMARY KEY,
            preset TEXT NOT NULL,
            sample_encoding_preset TEXT NOT NULL,
            signal_state_preset TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            decoder TEXT NOT NULL,
            git_branch TEXT,
            git_commit TEXT,
            number_of_sequential_frames INTEGER,
            black_level INTEGER,
            has_nonstandard_values BOOLEAN,
            capture_notes TEXT
        );
    )";
    if (sqlite3_exec(db, schema, nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return false;
    }
    std::string sql =
        "INSERT INTO cvbs_file (cvbs_file_id, preset, sample_encoding_preset, "
        "signal_state_preset, signal_type, decoder) VALUES (1, '" + preset + "', '" +
        encoding + "', '" + state + "', '" + signalType + "', 'cvbs-encode')";
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return false;
    }
    sqlite3_close(db);
    return true;
}

bool writeUniformCvbs(const std::string &path, size_t numSamples, int16_t value) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    const std::vector<int16_t> buf(numSamples, value);
    out.write(reinterpret_cast<const char *>(buf.data()),
              static_cast<std::streamsize>(buf.size() * sizeof(int16_t)));
    return out.good();
}

// ─── Test 1: original ABI decode loop on a black NTSC TBC fixture. ─────────
int testSyncDecodeBlackMono(const fs::path &dir) {
    const std::string tbc     = (dir / "blk.tbc").string();
    const std::string sidecar = (dir / "blk.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 6;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

    chd_video_info_t info;
    REQUIRE(chd_video_get_info(video, &info) == CHD_OK);
    REQUIRE(info.num_frames == 3);

    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_MONO, &dec) == CHD_OK);

    REQUIRE(chd_decoder_has_option(dec, CHD_OPT_LUMA_NR_LEVEL) == CHD_OK);
    REQUIRE(chd_decoder_has_option(dec, CHD_OPT_CHROMA_NR_LEVEL) == CHD_E_INVALID_ARG);
    REQUIRE(chd_decoder_set_option_f64(dec, CHD_OPT_LUMA_NR_LEVEL, 0.0) == CHD_OK);
    REQUIRE(chd_decoder_set_option_f64(dec, CHD_OPT_CHROMA_NR_LEVEL, 0.0) == CHD_E_INVALID_ARG);
    REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);

    // Adaptive-3D knobs apply only to the adaptive 3D kind; on the other comb
    // kinds they are rejected, not silently ignored.
    {
        chd_decoder_t *ntsc2d = nullptr;
        chd_decoder_t *ntsc3d = nullptr;
        REQUIRE(chd_decoder_create(video, CHD_DEC_NTSC_2D, &ntsc2d) == CHD_OK);
        REQUIRE(chd_decoder_create(video, CHD_DEC_NTSC_3D, &ntsc3d) == CHD_OK);
        for (const char *opt : {CHD_OPT_COMB_ADAPT_THRESHOLD,
                                CHD_OPT_COMB_CHROMA_WEIGHT,
                                CHD_OPT_COMB_SHOW_MAP}) {
            REQUIRE(chd_decoder_has_option(ntsc3d, opt) == CHD_OK);
            REQUIRE(chd_decoder_has_option(ntsc2d, opt) == CHD_E_INVALID_ARG);
        }
        chd_decoder_free(ntsc2d);
        chd_decoder_free(ntsc3d);
    }

    {
        // A kind that names no enumerator is still a valid value of the enum
        // type itself (fixed int32_t underlying type): accepted at create,
        // rejected at commit.
        chd_decoder_t *unknown = nullptr;
        REQUIRE(chd_decoder_create(video, static_cast<chd_decoder_kind_t>(99), &unknown) == CHD_OK);
        REQUIRE(chd_decoder_commit(unknown) == CHD_E_DECODER_UNKNOWN);
        chd_decoder_free(unknown);
    }

    chd_frame_t *bad = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &bad) == CHD_E_INVALID_ARG);
    REQUIRE(bad == nullptr);

    REQUIRE(chd_decoder_commit(dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_f64(dec, CHD_OPT_LUMA_NR_LEVEL, 0.5) == CHD_E_INVALID_ARG);

    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    REQUIRE(frame != nullptr);

    chd_frame_info_t finfo;
    REQUIRE(chd_frame_get_info(frame, &finfo) == CHD_OK);
    REQUIRE(finfo.format == CHD_PIXEL_YUV444P16);
    REQUIRE(finfo.frame_index == 0);
    REQUIRE(finfo.num_planes == 3);

    const void *yData = nullptr;
    ptrdiff_t yStride = 0;
    REQUIRE(chd_frame_get_plane(frame, CHD_PLANE_Y, &yData, &yStride) == CHD_OK);
    const uint16_t *yRow = static_cast<const uint16_t *>(yData);
    REQUIRE(yRow[0] == 4096);
    REQUIRE(yRow[finfo.width / 2] == 4096);

    const void *cbData = nullptr;
    ptrdiff_t cbStride = 0;
    REQUIRE(chd_frame_get_plane(frame, CHD_PLANE_CB, &cbData, &cbStride) == CHD_OK);
    REQUIRE(static_cast<const uint16_t *>(cbData)[0] == 32768);

    chd_frame_free(frame);

    chd_frame_t *oob = nullptr;
    REQUIRE(chd_decode_frame(dec, 99, &oob) == CHD_E_OUT_OF_RANGE);
    REQUIRE(oob == nullptr);

    // Async + cancel
    struct AsyncSink {
        std::mutex                   mu;
        std::vector<chd_status_t>    statuses;
        std::vector<chd_frame_t *>   frames;
    } sink;
    auto cb = +[](void *user, chd_status_t s, int64_t /*idx*/, chd_frame_t *f) {
        auto *sk = static_cast<AsyncSink *>(user);
        std::lock_guard<std::mutex> lock(sk->mu);
        sk->statuses.push_back(s);
        sk->frames.push_back(f);
    };
    const int64_t idxs[2] = {0, 1};
    REQUIRE(chd_decode_frames_async(dec, idxs, 2, cb, &sink, nullptr) == CHD_OK);
    REQUIRE(sink.statuses.size() == 2);
    for (size_t i = 0; i < 2; i++) {
        REQUIRE(sink.statuses[i] == CHD_OK);
        REQUIRE(sink.frames[i] != nullptr);
        chd_frame_free(sink.frames[i]);
    }
    sink.statuses.clear();
    sink.frames.clear();

    chd_cancel_t *cancel = nullptr;
    REQUIRE(chd_cancel_create(&cancel) == CHD_OK);
    chd_cancel_request(cancel);
    REQUIRE(chd_cancel_is_requested(cancel) == 1);
    REQUIRE(chd_decode_frames_async(dec, idxs, 2, cb, &sink, cancel) == CHD_OK);
    REQUIRE(sink.statuses.size() == 2);
    for (size_t i = 0; i < 2; i++) {
        REQUIRE(sink.statuses[i] == CHD_E_CANCELLED);
        REQUIRE(sink.frames[i] == nullptr);
    }
    chd_cancel_free(cancel);

    chd_decoder_free(dec);
    chd_video_free(video);

    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test 2 (hardening): CVBS primary + TBC extra source. ──────────────────
//
// Previously, the TBC extra-source path rejected this
// case with "primary source has no TBC metadata" because CVBS opens left
// v->metadata == nullptr. Hardening synthesizes metadata at open time and
// drops the check.
int testCvbsPrimaryWithTbcExtra(const fs::path &dir) {
    const std::string composite  = (dir / "cvbs.cvbs").string();
    const std::string compositeM = (dir / "cvbs.meta").string();
    const std::string extraTbc   = (dir / "extra.tbc").string();
    const std::string extraDb    = (dir / "extra.tbc.db").string();

    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    // CVBS primary: 4 NTSC fields, uniform blanking at 256 (CVBS_U10_4FSC
    // ⇒ × 64 ⇒ 16384 which matches the TBC blanking16bIre default).
    REQUIRE(writeUniformCvbs(composite,
                             static_cast<size_t>(fieldWidth) * fieldHeight * numFields,
                             256));
    REQUIRE(writeCvbsMeta(compositeM, "NTSC", "CVBS_U10_4FSC",
                          "STANDARD_TBC_LOCKED", "composite"));

    // TBC extra
    REQUIRE(writeBlackTbc(extraTbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(extraDb, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), compositeM.c_str(),
                                          nullptr, &video) == CHD_OK);

    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(video, &info) == CHD_OK);
    // A CVBS source reports the actual sample encoding it was opened with.
    REQUIRE(info.encoding == CHD_ENC_CVBS_U10_4FSC);
    REQUIRE(info.num_frames == 2);  // 4 fields / 2

    // This used to fail with "primary source has no TBC metadata".
    REQUIRE(chd_video_add_extra_source_composite(video, extraTbc.c_str(), nullptr) == CHD_OK);

    chd_video_free(video);

    fs::remove(composite);
    fs::remove(compositeM);
    fs::remove(extraTbc);
    fs::remove(extraDb);
    return 0;
}

// ─── Test 3 (hardening): multi-source dropout actually fires. ──────────────
//
// Inject a drop_outs row on a specific line of the primary's first field.
// Add an extra TBC source. Decode with chd_dropout_opts.enabled=1. Expect
// chd_decoder_get_last_dropout_stats to show corrected ≥ 1.
int testMultiSourceDropoutAbi(const fs::path &dir) {
    const std::string primTbc = (dir / "prim.tbc").string();
    const std::string primDb  = (dir / "prim.tbc.db").string();
    const std::string exTbc   = (dir / "ex.tbc").string();
    const std::string exDb    = (dir / "ex.tbc.db").string();

    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    REQUIRE(writeBlackTbc(primTbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeBlackTbc(exTbc,   fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(exDb, numFields));

    // Drop on first field (field_id = 0 in the 0-based seqNo column), line
    // 100, samples [200, 280). Inside the active region [147, 905).
    const DropoutRow drop{/*fieldId=*/0, /*startx=*/200, /*endx=*/280, /*fieldLine=*/100};
    REQUIRE(writeTbcSidecar(primDb, numFields, {drop}));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(primTbc.c_str(), primDb.c_str(), nullptr, &video) == CHD_OK);
    REQUIRE(chd_video_add_extra_source_composite(video, exTbc.c_str(), nullptr) == CHD_OK);

    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_MONO, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);

    chd_dropout_opts_t dops{};
    dops.enabled = 1;
    dops.overcorrect = 0;
    dops.intra_field_only = 0;
    REQUIRE(chd_decoder_set_dropout(dec, &dops) == CHD_OK);

    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    REQUIRE(frame != nullptr);

    chd_dropout_stats_t stats{};
    REQUIRE(chd_decoder_get_last_dropout_stats(dec, &stats) == CHD_OK);
    // One drop_outs row → at least one corrected dropout region. failed
    // should be zero since the extra has the same line cleanly.
    REQUIRE(stats.corrected >= 1);
    REQUIRE(stats.failed == 0);

    chd_frame_free(frame);
    chd_decoder_free(dec);
    chd_video_free(video);

    fs::remove(primTbc);
    fs::remove(primDb);
    fs::remove(exTbc);
    fs::remove(exDb);
    return 0;
}

// ─── Test 4 (hardening follow-up): parallel async dispatch. ────────────────
//
// thread_count=4, 12 indices. Each worker owns its own decoder instance,
// pulls next-index from an atomic counter, fires the callback with its
// own frame. We verify all 12 callbacks land with CHD_OK and a valid Y
// plane (black point Y_ZERO=4096). The result set must cover every
// requested index exactly once — proves the atomic counter doesn't
// double-dispatch and the workers actually drain the queue.
int testParallelAsyncDispatch(const fs::path &dir) {
    const std::string tbc     = (dir / "par.tbc").string();
    const std::string sidecar = (dir / "par.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFrames   = 16;
    constexpr int32_t numFields   = numFrames * 2;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_MONO, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);
    // Force a concrete worker count so the test exercises the
    // multi-worker dispatch (thread_count=0 would also work but its
    // effective W depends on std::thread::hardware_concurrency()).
    REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_THREAD_COUNT, 4) == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    constexpr size_t N = 12;
    std::vector<int64_t> idxs(N);
    for (size_t i = 0; i < N; i++) idxs[i] = static_cast<int64_t>(i);

    struct Sink {
        std::mutex                 mu;
        std::vector<int64_t>       returned;
        std::vector<chd_status_t>  statuses;
        std::vector<chd_frame_t *> frames;
    } sink;

    auto cb = +[](void *user, chd_status_t s, int64_t idx, chd_frame_t *f) {
        auto *sk = static_cast<Sink *>(user);
        std::lock_guard<std::mutex> lock(sk->mu);
        sk->returned.push_back(idx);
        sk->statuses.push_back(s);
        sk->frames.push_back(f);
    };

    REQUIRE(chd_decode_frames_async(dec, idxs.data(), N, cb, &sink, nullptr) == CHD_OK);

    REQUIRE(sink.returned.size() == N);
    // Build a presence vector keyed by returned frame index; the order of
    // callbacks is not deterministic across W workers but the set must
    // exactly cover [0, N).
    std::vector<int> seen(N, 0);
    for (size_t i = 0; i < N; i++) {
        REQUIRE(sink.statuses[i] == CHD_OK);
        REQUIRE(sink.frames[i] != nullptr);
        const int64_t idx = sink.returned[i];
        REQUIRE(idx >= 0);
        REQUIRE(idx < static_cast<int64_t>(N));
        seen[idx]++;

        // Spot-check each frame's Y plane: black input ⇒ Y_ZERO=4096
        // across every pixel.
        chd_frame_info_t finfo{};
        REQUIRE(chd_frame_get_info(sink.frames[i], &finfo) == CHD_OK);
        REQUIRE(finfo.frame_index == idx);
        const void *yData = nullptr;
        ptrdiff_t yStride = 0;
        REQUIRE(chd_frame_get_plane(sink.frames[i], CHD_PLANE_Y, &yData, &yStride) == CHD_OK);
        const uint16_t *yRow = static_cast<const uint16_t *>(yData);
        REQUIRE(yRow[0] == 4096);
        REQUIRE(yRow[finfo.width / 2] == 4096);

        chd_frame_free(sink.frames[i]);
    }
    for (size_t i = 0; i < N; i++) REQUIRE(seen[i] == 1);

    chd_decoder_free(dec);
    chd_video_free(video);

    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

}  // namespace

// ─── Test 6: tbc-tools v4 active-line columns. ─────────────────────────────
//
// tbc-tools added first/last_active_field_line + first/last_active_frame_line
// columns to the capture table at schema v4 (commit a0f45b0). Our SQLite reader
// feeds the frame-line columns through LineParameters::applyTo so the resulting
// VideoParameters override the standard ld-decode defaults; the field-line
// columns are tolerated when present but not read (the active field-line range
// is derived from the frame-line crop). Caller-supplied
// CHD_OPT_*_ACTIVE_FRAME_LINE options should still win over sidecar values,
// mirroring the established option precedence.

// tbc-tools v4 capture schema (only the columns we need for this test —
// PRAGMA table_info detection picks up the optional ones we set).
const char *kTbcSchemaV4 = R"(
PRAGMA user_version = 4;

CREATE TABLE IF NOT EXISTS capture (
    capture_id INTEGER PRIMARY KEY,
    system TEXT NOT NULL,
    decoder TEXT NOT NULL,
    git_branch TEXT,
    git_commit TEXT,
    video_sample_rate REAL,
    active_video_start INTEGER,
    active_video_end INTEGER,
    field_width INTEGER,
    field_height INTEGER,
    number_of_sequential_fields INTEGER,
    colour_burst_start INTEGER,
    colour_burst_end INTEGER,
    is_mapped INTEGER,
    is_subcarrier_locked INTEGER,
    is_widescreen INTEGER,
    white_16b_ire INTEGER,
    black_16b_ire INTEGER,
    blanking_16b_ire INTEGER,
    first_active_field_line INTEGER,
    last_active_field_line INTEGER,
    first_active_frame_line INTEGER,
    last_active_frame_line INTEGER,
    capture_notes TEXT
);

CREATE TABLE IF NOT EXISTS field_record (
    capture_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    audio_samples INTEGER,
    decode_faults INTEGER,
    disk_loc REAL,
    efm_t_values INTEGER,
    field_phase_id INTEGER,
    file_loc INTEGER,
    is_first_field INTEGER,
    median_burst_ire REAL,
    pad INTEGER,
    sync_conf INTEGER,
    ntsc_is_fm_code_data_valid INTEGER,
    ntsc_fm_code_data INTEGER,
    ntsc_field_flag INTEGER,
    ntsc_is_video_id_data_valid INTEGER,
    ntsc_video_id_data INTEGER,
    ntsc_white_flag INTEGER
);
)";

bool writeTbcSidecarV4(const std::string &path, int32_t numFields,
                       int32_t firstFieldLine, int32_t lastFieldLine,
                       int32_t firstFrameLine, int32_t lastFrameLine) {
    if (fs::exists(path)) fs::remove(path);
    sqlite3 *db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) return false;
    if (sqlite3_exec(db, kTbcSchemaV4, nullptr, nullptr, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return false;
    }
    const char *capSql =
        "INSERT INTO capture (capture_id, system, decoder, video_sample_rate, "
        "  active_video_start, active_video_end, field_width, field_height, "
        "  number_of_sequential_fields, colour_burst_start, colour_burst_end, "
        "  is_mapped, is_subcarrier_locked, is_widescreen, "
        "  white_16b_ire, black_16b_ire, blanking_16b_ire, "
        "  first_active_field_line, last_active_field_line, "
        "  first_active_frame_line, last_active_frame_line) "
        "VALUES (1, 'NTSC', 'tbc-tools', 14318181.818, 147, 905, 910, 263, ?, "
        "        92, 119, 1, 1, 0, 51200, 17920, 16384, ?, ?, ?, ?);";
    sqlite3_stmt *cap = nullptr;
    if (sqlite3_prepare_v2(db, capSql, -1, &cap, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        return false;
    }
    sqlite3_bind_int(cap, 1, numFields);
    sqlite3_bind_int(cap, 2, firstFieldLine);
    sqlite3_bind_int(cap, 3, lastFieldLine);
    sqlite3_bind_int(cap, 4, firstFrameLine);
    sqlite3_bind_int(cap, 5, lastFrameLine);
    if (sqlite3_step(cap) != SQLITE_DONE) {
        sqlite3_finalize(cap);
        sqlite3_close(db);
        return false;
    }
    sqlite3_finalize(cap);

    const char *fieldSql =
        "INSERT INTO field_record (capture_id, field_id, is_first_field, "
        "  sync_conf, median_burst_ire, ntsc_is_fm_code_data_valid, "
        "  ntsc_fm_code_data, ntsc_field_flag, ntsc_is_video_id_data_valid, "
        "  ntsc_video_id_data, ntsc_white_flag) "
        "VALUES (1, ?, ?, 100, 25.0, 0, 0, 0, 0, 0, 0);";
    sqlite3_stmt *fs = nullptr;
    sqlite3_prepare_v2(db, fieldSql, -1, &fs, nullptr);
    for (int32_t i = 0; i < numFields; i++) {
        sqlite3_reset(fs);
        sqlite3_bind_int(fs, 1, i);
        sqlite3_bind_int(fs, 2, (i % 2 == 0) ? 1 : 0);
        if (sqlite3_step(fs) != SQLITE_DONE) {
            sqlite3_finalize(fs);
            sqlite3_close(db);
            return false;
        }
    }
    sqlite3_finalize(fs);
    sqlite3_close(db);
    return true;
}

// Decode one frame against the given video + decoder; return the
// chd_frame_info_t::height of frame 0. Used as a probe for active-line
// settings: output height = (lastActiveFrameLine - firstActiveFrameLine)
// after padding, so changes in line range surface as height changes.
int32_t probeFrameHeight(chd_video_t *video, int32_t paddingMultiple,
                         const std::vector<std::pair<const char *, int32_t>> &i32Options) {
    chd_decoder_t *dec = nullptr;
    if (chd_decoder_create(video, CHD_DEC_MONO, &dec) != CHD_OK) return -1;
    if (chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, paddingMultiple) != CHD_OK) {
        chd_decoder_free(dec); return -1;
    }
    for (const auto &opt : i32Options) {
        if (chd_decoder_set_option_i32(dec, opt.first, opt.second) != CHD_OK) {
            chd_decoder_free(dec); return -1;
        }
    }
    if (chd_decoder_commit(dec) != CHD_OK) { chd_decoder_free(dec); return -1; }
    chd_frame_t *frame = nullptr;
    if (chd_decode_frame(dec, 0, &frame) != CHD_OK || frame == nullptr) {
        chd_decoder_free(dec); return -1;
    }
    chd_frame_info_t finfo;
    chd_frame_get_info(frame, &finfo);
    const int32_t h = finfo.height;
    chd_frame_free(frame);
    chd_decoder_free(dec);
    return h;
}

int testActiveLineSidecarOverride(const fs::path &dir) {
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    auto runProbe = [&](const std::string &tbc, const std::string &sidecar,
                        bool useV4, int32_t expectedHeight,
                        const std::vector<std::pair<const char *, int32_t>> &i32Options) {
        // Remove leftover state from any prior failing run — sqlite_open
        // keeps existing files, so a leftover capture row would hit a
        // UNIQUE-constraint failure on insert.
        if (fs::exists(tbc))     fs::remove(tbc);
        if (fs::exists(sidecar)) fs::remove(sidecar);

        REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
        if (useV4) {
            REQUIRE(writeTbcSidecarV4(sidecar, numFields, /*ffl=*/40, /*lfl=*/241,
                                                          /*ffrl=*/80, /*lfrl=*/480));
        } else {
            REQUIRE(writeTbcSidecar(sidecar, numFields));
        }

        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);
        const int32_t h = probeFrameHeight(video, /*padding=*/1, i32Options);
        REQUIRE(h == expectedHeight);
        chd_video_free(video);
        fs::remove(tbc);
        fs::remove(sidecar);
        return 0;
    };

    // Baseline: standard ld-decode sidecar (no v4 columns). NTSC default
    // active frame lines are 39..524 inclusive → height = 486 with padding=1.
    if (int rc = runProbe((dir / "baseline.tbc").string(),
                          (dir / "baseline.tbc.db").string(),
                          /*useV4=*/false, /*expectedHeight=*/486, {});
        rc != 0) return rc;

    // tbc-tools v4 sidecar with custom frame lines ffrl=80, lfrl=480. tbc-tools
    // writes lfrl as an exclusive bound, so the SQLite reader translates it to
    // our inclusive scheme (480 → 479): active rows 80..479 → height = 400.
    // Verifies the reader picked up the columns and processed them through
    // LineParameters::applyTo with the convention translation applied.
    if (int rc = runProbe((dir / "v4.tbc").string(),
                          (dir / "v4.tbc.db").string(),
                          /*useV4=*/true, /*expectedHeight=*/400, {});
        rc != 0) return rc;

    // Same v4 sidecar (lfrl=480 → inclusive 479) but the caller explicitly sets
    // CHD_OPT_FIRST_ACTIVE_FRAME_LINE=40 (the C API option is already inclusive,
    // no translation) → active rows 40..479 → height = 440. Verifies caller-set
    // options override sidecar values.
    if (int rc = runProbe((dir / "v4_override.tbc").string(),
                          (dir / "v4_override.tbc.db").string(),
                          /*useV4=*/true, /*expectedHeight=*/440,
                          {{CHD_OPT_FIRST_ACTIVE_FRAME_LINE, 40}});
        rc != 0) return rc;

    return 0;
}

// ─── Test 5: CHD_OPT_REVERSE_FIELD_ORDER. ──────────────────────────────────
//
// Matches upstream ld-chroma-decoder `-r`: chd_decoder_commit flips the
// metadata's isFirstFieldFirst flag. The fixture has 6 sequential fields
// with isFirstField alternating (1,0,1,0,1,0). With the default flag,
// getNumberOfFrames() = 3 and is_first_field_first = 1. After commit with
// REVERSE_FIELD_ORDER, the flag flips and getNumberOfFrames() reports 2
// (field 0 is now treated as a leftover that doesn't open a still-frame),
// observable via a fresh chd_video_get_info call.
int testReverseFieldOrder(const fs::path &dir) {
    const std::string tbc     = (dir / "rev.tbc").string();
    const std::string sidecar = (dir / "rev.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 6;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

    chd_video_info_t info;
    REQUIRE(chd_video_get_info(video, &info) == CHD_OK);
    REQUIRE(info.num_frames == 3);
    REQUIRE(info.is_first_field_first == 1);

    // REVERSE_FIELD_ORDER is universally valid on all decoder kinds (it's
    // an output-side metadata option, not algorithm-specific).
    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_MONO, &dec) == CHD_OK);
    REQUIRE(chd_decoder_has_option(dec, CHD_OPT_REVERSE_FIELD_ORDER) == CHD_OK);
    REQUIRE(chd_decoder_set_option_bool(dec, CHD_OPT_REVERSE_FIELD_ORDER, 1) == CHD_OK);
    REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    // After commit, the metadata flag should be flipped. Both observable
    // effects: num_frames drops by 1 (the leftover field doesn't form a
    // frame) and is_first_field_first toggles.
    chd_video_info_t info2;
    REQUIRE(chd_video_get_info(video, &info2) == CHD_OK);
    REQUIRE(info2.num_frames == 2);
    REQUIRE(info2.is_first_field_first == 0);

    // Decode still succeeds against the reversed assignment (black either way).
    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    REQUIRE(frame != nullptr);
    chd_frame_free(frame);

    chd_decoder_free(dec);
    chd_video_free(video);
    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: active-line crop overrides reject bounds past the raster. ───────
//
// The frame-line escape-hatch bound feeds a buffer index and must be rejected
// at commit:
//
//   Frame lines index ComponentFrame, which allocates (fieldHeight * 2) - 1
//   rows (the second field's trailing line is padding, never stored). The last
//   valid frame line is (fieldHeight * 2) - 2; for NTSC (fieldHeight 263) that
//   is 524, and 525 is one row past the buffer.
int testActiveLineCropBounds(const fs::path &dir) {
    const std::string tbc     = (dir / "crop_bound.tbc").string();
    const std::string sidecar = (dir / "crop_bound.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

    // Commit a single-option override and report the status. Each case gets a
    // fresh uncommitted decoder.
    auto commitWith = [&](const char *opt, int32_t value) -> chd_status_t {
        chd_decoder_t *dec = nullptr;
        if (chd_decoder_create(video, CHD_DEC_MONO, &dec) != CHD_OK) return CHD_E_INVALID_ARG;
        const chd_status_t set = chd_decoder_set_option_i32(dec, opt, value);
        const chd_status_t rc  = (set == CHD_OK) ? chd_decoder_commit(dec) : set;
        chd_decoder_free(dec);
        return rc;
    };

    // The last stored frame line is accepted; one past it is rejected.
    REQUIRE(commitWith(CHD_OPT_LAST_ACTIVE_FRAME_LINE, (fieldHeight * 2) - 2) == CHD_OK);
    REQUIRE(commitWith(CHD_OPT_LAST_ACTIVE_FRAME_LINE, (fieldHeight * 2) - 1) == CHD_E_INVALID_ARG);

    chd_video_free(video);
    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: float output formats expose normalized E'Y E'Cb E'Cr. ───────────
//
// Black NTSC input → E'Y = 0.0 (black) and neutral chroma E'Cb = E'Cr = 0.0,
// the float counterparts of the integer path's Y_ZERO (4096) / C_ZERO (32768).
// Covers both float formats plus the get_plane / get_plane_float guard rails:
//   - yuv444ps: 3 float planes, integer get_plane refused
//   - grays:    1 float plane, Cb/Cr refused
int testFloatOutputFormats(const fs::path &dir) {
    const std::string tbc     = (dir / "flt.tbc").string();
    const std::string sidecar = (dir / "flt.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    // ── yuv444ps: three normalized E' planes ──
    {
        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);
        chd_decoder_t *dec = nullptr;
        REQUIRE(chd_decoder_create(video, CHD_DEC_NTSC_2D, &dec) == CHD_OK);
        REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);
        REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv444ps") == CHD_OK);
        REQUIRE(chd_decoder_commit(dec) == CHD_OK);

        chd_frame_t *frame = nullptr;
        REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
        REQUIRE(frame != nullptr);

        chd_frame_info_t finfo{};
        REQUIRE(chd_frame_get_info(frame, &finfo) == CHD_OK);
        REQUIRE(finfo.format == CHD_PIXEL_YUV444PS);
        REQUIRE(finfo.num_planes == 3);

        // Integer get_plane must refuse a float frame.
        const void *bogus = nullptr;
        ptrdiff_t bogusStride = 0;
        REQUIRE(chd_frame_get_plane(frame, CHD_PLANE_Y, &bogus, &bogusStride)
                == CHD_E_INVALID_ARG);

        const float *yData = nullptr;
        ptrdiff_t yStride = 0;
        REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_Y, &yData, &yStride) == CHD_OK);
        REQUIRE(yStride == static_cast<ptrdiff_t>(finfo.width) * static_cast<ptrdiff_t>(sizeof(float)));
        REQUIRE(yData[0] == 0.0f);              // black ⇒ E'Y = 0.0
        REQUIRE(yData[finfo.width / 2] == 0.0f);

        const float *cbData = nullptr;
        const float *crData = nullptr;
        ptrdiff_t cStride = 0;
        REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_CB, &cbData, &cStride) == CHD_OK);
        REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_CR, &crData, &cStride) == CHD_OK);
        REQUIRE(cbData[0] == 0.0f);             // neutral chroma ⇒ E'Cb = 0.0
        REQUIRE(crData[0] == 0.0f);

        chd_frame_free(frame);
        chd_decoder_free(dec);
        chd_video_free(video);
    }

    // ── grays: single normalized E'Y plane ──
    {
        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);
        chd_decoder_t *dec = nullptr;
        REQUIRE(chd_decoder_create(video, CHD_DEC_MONO, &dec) == CHD_OK);
        REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);
        REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "grays") == CHD_OK);
        REQUIRE(chd_decoder_commit(dec) == CHD_OK);

        chd_frame_t *frame = nullptr;
        REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
        REQUIRE(frame != nullptr);

        chd_frame_info_t finfo{};
        REQUIRE(chd_frame_get_info(frame, &finfo) == CHD_OK);
        REQUIRE(finfo.format == CHD_PIXEL_GRAYS);
        REQUIRE(finfo.num_planes == 1);

        const float *yData = nullptr;
        ptrdiff_t yStride = 0;
        REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_Y, &yData, &yStride) == CHD_OK);
        REQUIRE(yData[0] == 0.0f);

        // grays has no chroma planes.
        const float *cbData = nullptr;
        ptrdiff_t cbStride = 0;
        REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_CB, &cbData, &cbStride)
                == CHD_E_INVALID_ARG);

        chd_frame_free(frame);
        chd_decoder_free(dec);
        chd_video_free(video);
    }

    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: rgbs output produces normalized R'G'B' planes. ─────────────────
//
// Black NTSC input → E'Y = 0, neutral chroma → E'R = E'G = E'B = 0. Verifies
// the dedicated convertToFloatRGB path: 3 contiguous float planes addressable
// by CHD_PLANE_R/G/B (not Y/Cb/Cr), Y plane access refused.
int testRgbsOutputFormat(const fs::path &dir) {
    const std::string tbc     = (dir / "rgbs.tbc").string();
    const std::string sidecar = (dir / "rgbs.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);
    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_NTSC_2D, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "rgbs") == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    REQUIRE(frame != nullptr);

    chd_frame_info_t finfo{};
    REQUIRE(chd_frame_get_info(frame, &finfo) == CHD_OK);
    REQUIRE(finfo.format == CHD_PIXEL_RGBS);
    REQUIRE(finfo.num_planes == 3);

    // Y/Cb/Cr plane accessors should refuse an RGBS frame.
    const float *bogus = nullptr;
    ptrdiff_t bogusStride = 0;
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_Y, &bogus, &bogusStride)
            == CHD_E_INVALID_ARG);

    const float *rData = nullptr;
    const float *gData = nullptr;
    const float *bData = nullptr;
    ptrdiff_t stride = 0;
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_R, &rData, &stride) == CHD_OK);
    REQUIRE(stride == static_cast<ptrdiff_t>(finfo.width) * static_cast<ptrdiff_t>(sizeof(float)));
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_G, &gData, &stride) == CHD_OK);
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_B, &bData, &stride) == CHD_OK);
    REQUIRE(rData[0] == 0.0f);  // black + neutral chroma ⇒ E'R = 0
    REQUIRE(gData[0] == 0.0f);
    REQUIRE(bData[0] == 0.0f);

    // Integer get_plane must also refuse.
    const void *bogusInt = nullptr;
    REQUIRE(chd_frame_get_plane(frame, CHD_PLANE_R, &bogusInt, &bogusStride)
            == CHD_E_INVALID_ARG);

    chd_frame_free(frame);
    chd_decoder_free(dec);
    chd_video_free(video);

    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: output_clamp option accepts known tokens, rejects garbage. ─────
int testOutputClampOption(const fs::path &dir) {
    const std::string tbc     = (dir / "clamp.tbc").string();
    const std::string sidecar = (dir / "clamp.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    const char *const validTokens[] = {"none", "legal_rgb_sdr", "legal_rgb_hdr",
                                        "legal_ycbcr_bt601"};
    for (const char *token : validTokens) {
        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);
        chd_decoder_t *dec = nullptr;
        REQUIRE(chd_decoder_create(video, CHD_DEC_NTSC_2D, &dec) == CHD_OK);
        REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_CLAMP, token) == CHD_OK);
        REQUIRE(chd_decoder_commit(dec) == CHD_OK);
        chd_decoder_free(dec);
        chd_video_free(video);
    }

    // Unknown clamp token must be rejected at commit time.
    {
        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);
        chd_decoder_t *dec = nullptr;
        REQUIRE(chd_decoder_create(video, CHD_DEC_NTSC_2D, &dec) == CHD_OK);
        REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_CLAMP, "garbage") == CHD_OK);
        REQUIRE(chd_decoder_commit(dec) == CHD_E_INVALID_ARG);
        chd_decoder_free(dec);
        chd_video_free(video);
    }

    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: phase_compensation is accepted by the kinds that implement it. ──
//
// The comb kinds decode on the burst-locked axes; the ldzeug2 kinds rotate
// their demodulated output onto the measured phase. Kinds
// with no NTSC subcarrier to lock to must still refuse it. The ldzeug2 kinds
// need NN support compiled in, and reject the option along with everything
// else when they are unavailable, so only assert on them when the build has
// them.
int testPhaseCompensationOptionScope(const fs::path &dir) {
    const std::string tbc     = (dir / "phasecomp.tbc").string();
    const std::string sidecar = (dir / "phasecomp.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    auto accepts = [&](chd_decoder_kind_t kind, chd_status_t *setStatus) -> bool {
        chd_video_t *video = nullptr;
        if (chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) != CHD_OK) {
            return false;
        }
        chd_decoder_t *dec = nullptr;
        if (chd_decoder_create(video, kind, &dec) != CHD_OK) {
            chd_video_free(video);
            return false;
        }
        *setStatus = chd_decoder_set_option_bool(dec, CHD_OPT_PHASE_COMPENSATION, 1);
        chd_decoder_free(dec);
        chd_video_free(video);
        return true;
    };

    chd_status_t st = CHD_OK;

    REQUIRE(accepts(CHD_DEC_NTSC_2D, &st));
    REQUIRE(st == CHD_OK);

    // Mono has no chroma to phase-correct.
    REQUIRE(accepts(CHD_DEC_MONO, &st));
    REQUIRE(st != CHD_OK);

    // PAL's line-alternating V handles the same errors differently; this
    // option is NTSC-only.
    REQUIRE(accepts(CHD_DEC_PAL_2D, &st));
    REQUIRE(st != CHD_OK);

#if defined(CHD_WITH_NN)
    for (const chd_decoder_kind_t kind : {CHD_DEC_LDZEUG_COLOR_CNN,
                                          CHD_DEC_LDZEUG_LUMA_SEP,
                                          CHD_DEC_LDZEUG_LUMA_SEP_FRAME}) {
        REQUIRE(accepts(kind, &st));
        REQUIRE(st == CHD_OK);
    }
#endif

    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: decode-level Y/C merge from a luma.tbc + chroma.tbc pair. ───────
//
// chd_video_open_yc on an ld-decode luma + chroma .tbc pair takes the
// decode-merge path: the luma plane is Mono-decoded for Y and the chroma plane
// colour-decoded for U/V, then merged. (That open even succeeding proves the
// path engaged: a .tbc.db sidecar is ld-decode schema, so the CVBS YC path
// would fail to parse it.) Both planes are black NTSC here, so the result is
// black luma + neutral chroma — the point is to exercise the dual-source,
// dual-decoder wiring end to end.
int testYcMergeDualTbc(const fs::path &dir) {
    const std::string lumaTbc   = (dir / "luma.tbc").string();
    const std::string lumaDb    = (dir / "luma.tbc.db").string();
    const std::string chromaTbc = (dir / "chroma.tbc").string();
    const std::string chromaDb  = (dir / "chroma.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 6;

    REQUIRE(writeBlackTbc(lumaTbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(lumaDb, numFields));
    REQUIRE(writeBlackTbc(chromaTbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(chromaDb, numFields));

    chd_video_t *video = nullptr;
    // Luma sidecar passed explicitly; the chroma sidecar auto-locates next to
    // chroma.tbc. A .db (ld-decode) sidecar selects the decode-merge path.
    REQUIRE(chd_video_open_yc(lumaTbc.c_str(), chromaTbc.c_str(), lumaDb.c_str(),
                              nullptr, &video) == CHD_OK);

    chd_video_info_t info;
    REQUIRE(chd_video_get_info(video, &info) == CHD_OK);
    REQUIRE(info.num_frames == 3);
    REQUIRE(info.encoding == CHD_ENC_CVBS_U16_4FSC);

    // The configured colour kind decodes the chroma plane; luma is forced Mono.
    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_NTSC_2D, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1) == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    REQUIRE(frame != nullptr);

    chd_frame_info_t finfo;
    REQUIRE(chd_frame_get_info(frame, &finfo) == CHD_OK);
    REQUIRE(finfo.num_planes == 3);

    const void *yData = nullptr;
    ptrdiff_t yStride = 0;
    REQUIRE(chd_frame_get_plane(frame, CHD_PLANE_Y, &yData, &yStride) == CHD_OK);
    REQUIRE(static_cast<const uint16_t *>(yData)[0] == 4096);  // black luma

    const void *cbData = nullptr;
    ptrdiff_t cbStride = 0;
    REQUIRE(chd_frame_get_plane(frame, CHD_PLANE_CB, &cbData, &cbStride) == CHD_OK);
    // Constant black chroma plane → neutral Cb (allow a tiny filter epsilon).
    const int cb0 = static_cast<const uint16_t *>(cbData)[0];
    REQUIRE(cb0 >= 32768 - 256 && cb0 <= 32768 + 256);

    chd_frame_free(frame);
    chd_decoder_free(dec);
    chd_video_free(video);

    fs::remove(lumaTbc);
    fs::remove(lumaDb);
    fs::remove(chromaTbc);
    fs::remove(chromaDb);
    return 0;
}

// Padding surrounds the picture with a black border and never moves or alters
// it; the active-sample crop is the only thing that changes which signal
// samples reach the picture.
//
// The fixture is a uniform white field, so picture samples and border samples
// cannot be confused: a black fixture would decode to the very value the
// border is filled with.
int testPaddingAndSampleCrop(const fs::path &dir) {
    const std::string tbc     = (dir / "pad.tbc").string();
    const std::string sidecar = (dir / "pad.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;
    constexpr uint16_t kWhite = 51200;  // white_16b_ire from the sidecar

    // Y'CbCr black, the value the border is filled with.
    constexpr uint16_t kYBlack = 16 * 256;
    constexpr uint16_t kCZero  = 128 * 256;

    REQUIRE(writeUniformTbc(tbc, fieldWidth, fieldHeight, numFields, kWhite));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

    // The sidecar's crop is active_video_start=147, active_video_end=905, which
    // it stores exclusive. The ABI reports both bounds inclusive, so the last
    // active sample is 904.
    chd_video_info_t vinfo;
    REQUIRE(chd_video_get_info(video, &vinfo) == CHD_OK);
    REQUIRE(vinfo.first_active_sample == 147);
    REQUIRE(vinfo.last_active_sample  == 904);
    REQUIRE(vinfo.field_width         == 910);

    auto decodeWith = [&](const std::vector<std::pair<const char *, int32_t>> &opts,
                          chd_frame_t **out) -> chd_status_t {
        chd_decoder_t *dec = nullptr;
        if (chd_status_t s = chd_decoder_create(video, CHD_DEC_MONO, &dec); s != CHD_OK) return s;
        for (const auto &o : opts) {
            if (chd_status_t s = chd_decoder_set_option_i32(dec, o.first, o.second); s != CHD_OK) {
                chd_decoder_free(dec);
                return s;
            }
        }
        if (chd_status_t s = chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv444p16");
            s != CHD_OK) { chd_decoder_free(dec); return s; }
        if (chd_status_t s = chd_decoder_commit(dec); s != CHD_OK) {
            chd_decoder_free(dec);
            return s;
        }
        const chd_status_t s = chd_decode_frame(dec, 0, out);
        chd_decoder_free(dec);
        return s;
    };

    // The crop with no padding, as the reference picture.
    chd_frame_t *plain = nullptr;
    REQUIRE(decodeWith({{CHD_OPT_PADDING_MULTIPLE, 1}}, &plain) == CHD_OK);
    chd_frame_info_t pinfo;
    REQUIRE(chd_frame_get_info(plain, &pinfo) == CHD_OK);
    REQUIRE(pinfo.width  == 758);   // 904 - 147 + 1
    REQUIRE(pinfo.height == 486);

    // The same decode padded to a multiple of 16: 758 -> 768 (5 left, 5 right)
    // and 486 -> 496 (5 top, 5 bottom), the picture staying centred.
    chd_frame_t *padded = nullptr;
    REQUIRE(decodeWith({{CHD_OPT_PADDING_MULTIPLE, 16}}, &padded) == CHD_OK);
    chd_frame_info_t qinfo;
    REQUIRE(chd_frame_get_info(padded, &qinfo) == CHD_OK);
    REQUIRE(qinfo.width  == 768);
    REQUIRE(qinfo.height == 496);
    constexpr int32_t kLeftPad = 5;
    constexpr int32_t kTopPad  = 5;

    const uint16_t *pY = nullptr;  ptrdiff_t pStride = 0;
    const uint16_t *qY = nullptr;  ptrdiff_t qStride = 0;
    const uint16_t *qCb = nullptr; ptrdiff_t qCbStride = 0;
    REQUIRE(chd_frame_get_plane(plain,  CHD_PLANE_Y,  reinterpret_cast<const void **>(&pY),  &pStride) == CHD_OK);
    REQUIRE(chd_frame_get_plane(padded, CHD_PLANE_Y,  reinterpret_cast<const void **>(&qY),  &qStride) == CHD_OK);
    REQUIRE(chd_frame_get_plane(padded, CHD_PLANE_CB, reinterpret_cast<const void **>(&qCb), &qCbStride) == CHD_OK);
    const ptrdiff_t pRow  = pStride / 2;
    const ptrdiff_t qRow  = qStride / 2;
    const ptrdiff_t qcRow = qCbStride / 2;

    // The picture is white, so it cannot be mistaken for the black border.
    REQUIRE(pY[0] > kYBlack + 1000);

    // Every border sample is black luma and neutral chroma, on all four sides.
    for (int32_t y = 0; y < qinfo.height; y++) {
        const bool padRow = (y < kTopPad) || (y >= kTopPad + pinfo.height);
        for (int32_t x = 0; x < qinfo.width; x++) {
            const bool padCol = (x < kLeftPad) || (x >= kLeftPad + pinfo.width);
            if (!padRow && !padCol) continue;
            REQUIRE(qY[y * qRow + x] == kYBlack);
            REQUIRE(qCb[y * qcRow + x] == kCZero);
        }
    }

    // The picture inside the border is bit-identical to the unpadded decode:
    // padding grew the frame without disturbing a single picture sample.
    for (int32_t y = 0; y < pinfo.height; y++) {
        for (int32_t x = 0; x < pinfo.width; x++) {
            REQUIRE(qY[(y + kTopPad) * qRow + (x + kLeftPad)] == pY[y * pRow + x]);
        }
    }
    chd_frame_free(padded);
    chd_frame_free(plain);

    // Widening the sample crop is what admits signal outside the default active
    // window: it grows the picture itself, not the border. This sidecar's crop
    // is the picture centred in SMPTE ST 244's digital active line, which on a
    // 0H-aligned 910-sample row is samples 142-909; asking for those bounds
    // exactly yields the standard's full 768-sample digital active line.
    chd_frame_t *wide = nullptr;
    REQUIRE(decodeWith({{CHD_OPT_PADDING_MULTIPLE, 1},
                        {CHD_OPT_FIRST_ACTIVE_SAMPLE, 142},
                        {CHD_OPT_LAST_ACTIVE_SAMPLE,  909}},
                       &wide) == CHD_OK);
    chd_frame_info_t winfo;
    REQUIRE(chd_frame_get_info(wide, &winfo) == CHD_OK);
    REQUIRE(winfo.width  == 768);   // the ST 244 digital active line
    REQUIRE(winfo.height == 486);   // the other axis is untouched
    chd_frame_free(wide);

    // A bound read from chd_video_get_info and set straight back is a no-op:
    // both are inclusive, so the round-trip needs no adjustment.
    chd_frame_t *same = nullptr;
    REQUIRE(decodeWith({{CHD_OPT_PADDING_MULTIPLE, 1},
                        {CHD_OPT_FIRST_ACTIVE_SAMPLE, vinfo.first_active_sample},
                        {CHD_OPT_LAST_ACTIVE_SAMPLE,  vinfo.last_active_sample}},
                       &same) == CHD_OK);
    chd_frame_info_t sinfo;
    REQUIRE(chd_frame_get_info(same, &sinfo) == CHD_OK);
    REQUIRE(sinfo.width == 758);
    chd_frame_free(same);

    // A crop that leaves the stored row, or selects nothing, is rejected at
    // commit rather than reading past the end of a line.
    chd_frame_t *bad = nullptr;
    REQUIRE(decodeWith({{CHD_OPT_LAST_ACTIVE_SAMPLE, fieldWidth}}, &bad) == CHD_E_INVALID_ARG);
    REQUIRE(decodeWith({{CHD_OPT_FIRST_ACTIVE_SAMPLE, -1}}, &bad) == CHD_E_INVALID_ARG);
    REQUIRE(decodeWith({{CHD_OPT_FIRST_ACTIVE_SAMPLE, 500},
                        {CHD_OPT_LAST_ACTIVE_SAMPLE,  499}}, &bad) == CHD_E_INVALID_ARG);

    // A padding multiple large enough to overflow the frame size is rejected,
    // as is a nonsensical one.
    REQUIRE(decodeWith({{CHD_OPT_PADDING_MULTIPLE, 1 << 20}}, &bad) == CHD_E_INVALID_ARG);
    REQUIRE(decodeWith({{CHD_OPT_PADDING_MULTIPLE, 0}}, &bad) == CHD_E_INVALID_ARG);
    REQUIRE(decodeWith({{CHD_OPT_PADDING_MULTIPLE, -8}}, &bad) == CHD_E_INVALID_ARG);

    chd_video_free(video);
    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: source frame line <-> field-sequential signal line conversion. ──
//
// NTSC raster (fieldHeight 263): field 1 = even frame lines / signal 1..263,
// field 2 = odd frame lines / signal 264..525. The mapping is a bijection over
// frame lines 0..524 and is deliberately non-monotonic (adjacent frame lines
// jump fields), so the default active region's frame lines 39..524 map to
// signal lines 283..263 (first > last).
int testSignalLineConversion(const fs::path &dir) {
    const std::string tbc     = (dir / "siglines.tbc").string();
    const std::string sidecar = (dir / "siglines.tbc.db").string();
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
    REQUIRE(writeTbcSidecar(sidecar, numFields));

    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

    int32_t s = 0, f = 0;
    // Spot values across both fields, including the non-monotonic default crop.
    REQUIRE(chd_video_frame_line_to_signal_line(video, 0, &s)   == CHD_OK && s == 1);
    REQUIRE(chd_video_frame_line_to_signal_line(video, 1, &s)   == CHD_OK && s == 264);
    REQUIRE(chd_video_frame_line_to_signal_line(video, 39, &s)  == CHD_OK && s == 283);
    REQUIRE(chd_video_frame_line_to_signal_line(video, 523, &s) == CHD_OK && s == 525);
    REQUIRE(chd_video_frame_line_to_signal_line(video, 524, &s) == CHD_OK && s == 263);
    REQUIRE(chd_video_signal_line_to_frame_line(video, 283, &f) == CHD_OK && f == 39);
    REQUIRE(chd_video_signal_line_to_frame_line(video, 263, &f) == CHD_OK && f == 524);
    REQUIRE(chd_video_signal_line_to_frame_line(video, 264, &f) == CHD_OK && f == 1);

    // Round-trips over the whole raster.
    for (int32_t fl = 0; fl <= (2 * fieldHeight) - 2; fl++) {
        REQUIRE(chd_video_frame_line_to_signal_line(video, fl, &s) == CHD_OK);
        REQUIRE(chd_video_signal_line_to_frame_line(video, s, &f) == CHD_OK);
        REQUIRE(f == fl);
    }

    // Ranges: frame lines 0..(2H-2); signal lines 1..(2H-1).
    REQUIRE(chd_video_frame_line_to_signal_line(video, -1, &s)                   == CHD_E_OUT_OF_RANGE);
    REQUIRE(chd_video_frame_line_to_signal_line(video, (2 * fieldHeight) - 1, &s) == CHD_E_OUT_OF_RANGE);
    REQUIRE(chd_video_signal_line_to_frame_line(video, 0, &f)                    == CHD_E_OUT_OF_RANGE);
    REQUIRE(chd_video_signal_line_to_frame_line(video, 2 * fieldHeight, &f)      == CHD_E_OUT_OF_RANGE);
    REQUIRE(chd_video_signal_line_to_frame_line(video, (2 * fieldHeight) - 1, &f) == CHD_OK && f == 523);

    // Null-argument guards.
    REQUIRE(chd_video_frame_line_to_signal_line(nullptr, 0, &s)  == CHD_E_INVALID_ARG);
    REQUIRE(chd_video_signal_line_to_frame_line(video, 1, nullptr) == CHD_E_INVALID_ARG);

    chd_video_free(video);
    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

// ─── Test: interface-standard sample <-> stored-row sample conversion. ──
//
// The rotation follows the source's horizontal alignment. The synthetic NTSC
// sidecar marks the capture subcarrier-locked (blanking-start rows), putting
// ST 244 sample 0 at row sample 142 and the digital active line 0..767 at
// 142..909; with the lock flag cleared the raster is line-locked (sync-start
// rows) and the same region lands at 125..892. A sidecar-less PAL .cvbs
// covers the 625-line constants (177 sync-start, 187 blanking-start).
int testStandardSampleConversion(const fs::path &dir) {
    constexpr int32_t fieldWidth  = 910;
    constexpr int32_t fieldHeight = 263;
    constexpr int32_t numFields   = 4;

    int32_t r = 0, s = 0;

    // Blanking-start: the default synthetic sidecar is subcarrier-locked.
    {
        const std::string tbc     = (dir / "stdsamp_sc.tbc").string();
        const std::string sidecar = (dir / "stdsamp_sc.tbc.db").string();
        REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
        REQUIRE(writeTbcSidecar(sidecar, numFields));
        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

        REQUIRE(chd_video_standard_sample_to_row_sample(video, 0, &r)   == CHD_OK && r == 142);
        REQUIRE(chd_video_standard_sample_to_row_sample(video, 767, &r) == CHD_OK && r == 909);
        REQUIRE(chd_video_standard_sample_to_row_sample(video, 768, &r) == CHD_OK && r == 0);
        REQUIRE(chd_video_row_sample_to_standard_sample(video, 142, &s) == CHD_OK && s == 0);
        REQUIRE(chd_video_row_sample_to_standard_sample(video, 0, &s)   == CHD_OK && s == 768);

        // Round-trips over the whole row.
        for (int32_t stdSample = 0; stdSample < fieldWidth; stdSample++) {
            REQUIRE(chd_video_standard_sample_to_row_sample(video, stdSample, &r) == CHD_OK);
            REQUIRE(chd_video_row_sample_to_standard_sample(video, r, &s) == CHD_OK);
            REQUIRE(s == stdSample);
        }

        // Ranges + null guards.
        REQUIRE(chd_video_standard_sample_to_row_sample(video, -1, &r)         == CHD_E_OUT_OF_RANGE);
        REQUIRE(chd_video_standard_sample_to_row_sample(video, fieldWidth, &r) == CHD_E_OUT_OF_RANGE);
        REQUIRE(chd_video_row_sample_to_standard_sample(video, -1, &s)         == CHD_E_OUT_OF_RANGE);
        REQUIRE(chd_video_row_sample_to_standard_sample(video, fieldWidth, &s) == CHD_E_OUT_OF_RANGE);
        REQUIRE(chd_video_standard_sample_to_row_sample(nullptr, 0, &r)        == CHD_E_INVALID_ARG);
        REQUIRE(chd_video_row_sample_to_standard_sample(video, 0, nullptr)     == CHD_E_INVALID_ARG);

        chd_video_free(video);
        fs::remove(tbc);
        fs::remove(sidecar);
    }

    // Sync-start: same raster with the subcarrier-lock flag cleared.
    {
        const std::string tbc     = (dir / "stdsamp_ll.tbc").string();
        const std::string sidecar = (dir / "stdsamp_ll.tbc.db").string();
        REQUIRE(writeBlackTbc(tbc, fieldWidth, fieldHeight, numFields));
        REQUIRE(writeTbcSidecar(sidecar, numFields, {}, /*subcarrierLocked=*/0));
        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(tbc.c_str(), sidecar.c_str(), nullptr, &video) == CHD_OK);

        REQUIRE(chd_video_standard_sample_to_row_sample(video, 0, &r)   == CHD_OK && r == 125);
        REQUIRE(chd_video_standard_sample_to_row_sample(video, 767, &r) == CHD_OK && r == 892);
        REQUIRE(chd_video_row_sample_to_standard_sample(video, 0, &s)   == CHD_OK && s == 785);
        REQUIRE(chd_video_row_sample_to_standard_sample(video, 125, &s) == CHD_OK && s == 0);

        chd_video_free(video);
        fs::remove(tbc);
        fs::remove(sidecar);
    }

    // 625-line constants, both alignments, via a sidecar-less PAL .cvbs.
    for (int32_t scLocked = 0; scLocked <= 1; scLocked++) {
        const std::string composite = (dir / "stdsamp_pal.cvbs").string();
        REQUIRE(writeUniformTbc(composite, 1135, 313, numFields, 16384));

        chd_video_params_t params{};
        params.standard             = CHD_STD_PAL;
        params.encoding             = CHD_ENC_CVBS_U16_4FSC;
        params.signal_state         = CHD_SIG_STANDARD_TBC_UNLOCKED;
        params.is_subcarrier_locked = scLocked;

        chd_video_t *video = nullptr;
        REQUIRE(chd_video_open_composite(composite.c_str(), nullptr, &params, &video) == CHD_OK);
        REQUIRE(chd_video_standard_sample_to_row_sample(video, 0, &r) == CHD_OK);
        REQUIRE(r == (scLocked ? 187 : 177));
        REQUIRE(chd_video_standard_sample_to_row_sample(video, 947, &r) == CHD_OK);
        REQUIRE(r == (scLocked ? 1134 : 1124));
        chd_video_free(video);
        fs::remove(composite);
    }

    return 0;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "chd_phase_g_test";
    fs::create_directories(dir);

    if (int rc = testSyncDecodeBlackMono(dir);    rc != 0) return rc;
    if (int rc = testCvbsPrimaryWithTbcExtra(dir); rc != 0) return rc;
    if (int rc = testMultiSourceDropoutAbi(dir);  rc != 0) return rc;
    if (int rc = testParallelAsyncDispatch(dir);  rc != 0) return rc;
    if (int rc = testReverseFieldOrder(dir);          rc != 0) return rc;
    if (int rc = testActiveLineSidecarOverride(dir);  rc != 0) return rc;
    if (int rc = testActiveLineCropBounds(dir);       rc != 0) return rc;
    if (int rc = testFloatOutputFormats(dir);         rc != 0) return rc;
    if (int rc = testRgbsOutputFormat(dir);           rc != 0) return rc;
    if (int rc = testOutputClampOption(dir);          rc != 0) return rc;
    if (int rc = testPhaseCompensationOptionScope(dir); rc != 0) return rc;
    if (int rc = testYcMergeDualTbc(dir);             rc != 0) return rc;
    if (int rc = testPaddingAndSampleCrop(dir);       rc != 0) return rc;
    if (int rc = testSignalLineConversion(dir);       rc != 0) return rc;
    if (int rc = testStandardSampleConversion(dir);   rc != 0) return rc;

    fs::remove(dir);
    std::cout << "test_decode_frame_abi: PASS\n";
    return 0;
}
