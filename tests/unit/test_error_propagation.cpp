// SPDX-License-Identifier: GPL-3.0-or-later
//
// Failures reach the caller through the return path, carrying the reason the
// layer that detected them recorded. Covers:
//
//   1. chd::log::fail() records chd_last_error() detail whether or not a
//      diagnostic sink is listening.
//   2. chd_video_open_* distinguishes missing file / unreadable samples /
//      corrupt sidecar in both the status code and the detail string.
//   3. Comb rejects geometry its fixed-size frame buffers cannot hold instead
//      of configuring itself and overflowing them during the decode.
//   4. Out-of-range field-metadata requests yield empty metadata rather than
//      indexing outside the field vector.
//   5. A failure that reaches the caller is marked CHD_LOG_F_RETURNED on the
//      diagnostic channel, so a consumer surfacing the status can drop the
//      duplicate rather than report the same reason twice.

#include <chromadec/chromadec.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "../../src/common/log.h"
#include "../../src/decoders/comb/ntsc_decoder.h"
#include "../../src/metadata/core.h"

namespace {

namespace fs = std::filesystem;

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond \
                  << " (last_error: " << chd_last_error() << ")\n"; \
        return 1; \
    } \
} while (0)

bool contains(const std::string &haystack, const std::string &needle) {
    return haystack.find(needle) != std::string::npos;
}

// A minimal ld-decode source: two NTSC fields of silence plus a JSON sidecar
// describing them. `declaredFields` is what the sidecar claims, so a value
// that disagrees with the fields array produces a corrupt sidecar.
void writeSource(const fs::path &tbc, int declaredFields, int fieldCount) {
    constexpr int kFieldWidth = 910;
    constexpr int kFieldHeight = 263;
    {
        std::ofstream out(tbc, std::ios::binary);
        const std::vector<char> zeros(kFieldWidth * kFieldHeight * 2 * 2, 0);
        out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    std::ofstream json(tbc.string() + ".json");
    json << R"({"videoParameters":{"numberOfSequentialFields":)" << declaredFields
         << R"(,"system":"NTSC","fieldWidth":910,"fieldHeight":263,"isSourcePal":false,)"
         << R"("black16bIre":16384,"white16bIre":54016,"fsc":0,"colourBurstStart":98,)"
         << R"("colourBurstEnd":110,"activeVideoStart":134,"activeVideoEnd":894,)"
         << R"("isSubcarrierLocked":true},"fields":[)";
    for (int i = 1; i <= fieldCount; i++) {
        if (i > 1) json << ",";
        json << R"({"seqNo":)" << i << R"(,"isFirstField":)" << ((i % 2) ? "true" : "false")
             << R"(,"syncConf":100,"medianBurstIRE":20,"fieldPhaseID":)" << i << "}";
    }
    json << "]}";
}

int testFailRecordsDetailWithoutSink() {
    // No sink installed: the diagnostic goes nowhere, but the detail must still
    // be there for the caller that is about to be handed a failure.
    chd_set_log_callback(nullptr, nullptr);
    chd_clear_last_error();
    REQUIRE(chd_log_is_enabled(CHD_LOG_ERROR) == 0);

    chd::log::fail() << "something" << "went wrong";
    REQUIRE(std::string(chd_last_error()) == "something went wrong");

    // error(), by contrast, is diagnostic only and leaves the detail alone.
    chd_clear_last_error();
    chd::log::error() << "not a returned failure";
    REQUIRE(std::string(chd_last_error()).empty());
    return 0;
}

int testVideoOpenStatuses(const fs::path &dir) {
    chd_video_t *v = nullptr;

    // Missing file.
    const auto missing = dir / "does-not-exist.tbc";
    chd_status_t rc = chd_video_open_composite(missing.string().c_str(), nullptr, nullptr, &v);
    REQUIRE(rc == CHD_E_FILE_NOT_FOUND);
    REQUIRE(v == nullptr);
    REQUIRE(contains(chd_last_error(), "does-not-exist.tbc"));

    // Sidecar claiming more fields than it lists: corrupt, not merely invalid,
    // and the detail names the inconsistency rather than "failed to read".
    const auto corrupt = dir / "corrupt.tbc";
    writeSource(corrupt, 9, 2);
    rc = chd_video_open_composite(corrupt.string().c_str(), nullptr, nullptr, &v);
    REQUIRE(rc == CHD_E_METADATA_CORRUPT);
    REQUIRE(v == nullptr);
    REQUIRE(contains(chd_last_error(), "numberOfSequentialFields"));

    // A well-formed sidecar whose sample file cannot be read is an I/O
    // failure, distinct from both of the above.
    const auto unreadable = dir / "unreadable.tbc";
    writeSource(unreadable, 2, 2);
    fs::permissions(unreadable, fs::perms::none);
    rc = chd_video_open_composite(unreadable.string().c_str(), nullptr, nullptr, &v);
    fs::permissions(unreadable, fs::perms::owner_read | fs::perms::owner_write);
    // Running as root defeats the permission bits; skip rather than fail.
    if (rc != CHD_OK) {
        REQUIRE(rc == CHD_E_IO);
        REQUIRE(v == nullptr);
        REQUIRE(contains(chd_last_error(), "unreadable.tbc"));
    } else {
        chd_video_free(v);
        v = nullptr;
    }

    // The good one still opens, and leaves no stale detail behind it.
    const auto good = dir / "good.tbc";
    writeSource(good, 2, 2);
    rc = chd_video_open_composite(good.string().c_str(), nullptr, nullptr, &v);
    REQUIRE(rc == CHD_OK);
    REQUIRE(v != nullptr);
    chd_video_free(v);
    return 0;
}

// The duplicate-reporting case a consumer hits: a corrupt sidecar produces one
// reason, and it arrives twice - once as the status detail, once at error level
// on the sink. CHD_LOG_F_RETURNED is what tells the second copy apart from an
// error that has no other way of reaching anyone.
struct FlagCapture {
    std::mutex mutex;
    std::vector<std::pair<chd_log_flags_t, std::string>> errors;
};

void flagCaptureSink(chd_log_level_t level, chd_log_flags_t flags, const char *message,
                     void *user_data) {
    if (level != CHD_LOG_ERROR) return;
    auto *capture = static_cast<FlagCapture *>(user_data);
    std::lock_guard<std::mutex> lock(capture->mutex);
    capture->errors.emplace_back(flags, message);
}

int testReturnedFailuresAreMarked(const fs::path &dir) {
    FlagCapture capture;
    chd_set_log_callback(flagCaptureSink, &capture);
    chd_set_log_level(CHD_LOG_DEBUG);
    chd_clear_last_error();

    const auto corrupt = dir / "flagged.tbc";
    writeSource(corrupt, 9, 2);
    chd_video_t *v = nullptr;
    const chd_status_t rc =
        chd_video_open_composite(corrupt.string().c_str(), nullptr, nullptr, &v);
    // Uninstall before asserting, so a failing REQUIRE cannot leave the sink
    // pointing at a capture that is about to go out of scope.
    chd_set_log_callback(nullptr, nullptr);

    REQUIRE(rc == CHD_E_METADATA_CORRUPT);
    REQUIRE(v == nullptr);
    const std::string detail = chd_last_error();

    int flagged = 0;
    for (const auto &[flags, message] : capture.errors) {
        if ((flags & CHD_LOG_F_RETURNED) == 0) continue;
        flagged++;
        // The flagged text is what the caller reads back, modulo the
        // entry-point prefix the ABI shim adds on the way out. That identity is
        // the whole point: it is what makes dropping the copy safe.
        REQUIRE(contains(detail, message));
    }
    REQUIRE(flagged == 1);

    chd_set_log_level(CHD_LOG_INFO);
    return 0;
}

chd::metadata::LdDecodeMetaData::VideoParameters makeNtscVp() {
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::NTSC;
    vp.fieldWidth  = 910;
    vp.fieldHeight = 263;
    vp.fSC = 315.0e6 / 88.0;
    vp.sampleRate = vp.fSC * 4.0;
    vp.activeVideoStart = 134;
    vp.activeVideoEnd   = 894;
    vp.firstActiveFrameLine = 39;
    vp.lastActiveFrameLine  = 524;
    vp.black16bIre    = 16128;
    vp.white16bIre    = 51200;
    vp.blanking16bIre = 15360;
    vp.isValid = true;
    return vp;
}

int testCombRejectsOversizedGeometry() {
    chd::decoders::comb::Comb::Configuration combConfig;

    {
        chd::decoders::comb::NtscDecoder decoder(combConfig);
        REQUIRE(decoder.configure(makeNtscVp()));
    }

    // Wider than Comb's fixed frame buffers: configuring would overflow them
    // once decoding started, so configure must fail and say why.
    {
        chd::decoders::comb::NtscDecoder decoder(combConfig);
        auto vp = makeNtscVp();
        vp.fieldWidth = 4096;
        chd_clear_last_error();
        REQUIRE(!decoder.configure(vp));
        REQUIRE(contains(chd_last_error(), "frame width"));
    }

    // Taller than the buffers.
    {
        chd::decoders::comb::NtscDecoder decoder(combConfig);
        auto vp = makeNtscVp();
        vp.fieldHeight = 2048;
        chd_clear_last_error();
        REQUIRE(!decoder.configure(vp));
        REQUIRE(contains(chd_last_error(), "frame height"));
    }

    // The comb filter reaches 16 samples behind activeVideoStart.
    {
        chd::decoders::comb::NtscDecoder decoder(combConfig);
        auto vp = makeNtscVp();
        vp.activeVideoStart = 4;
        chd_clear_last_error();
        REQUIRE(!decoder.configure(vp));
        REQUIRE(contains(chd_last_error(), "activeVideoStart"));
    }

    // A non-NTSC source is rejected by name, not by the generic guess the ABI
    // used to substitute.
    {
        chd::decoders::comb::NtscDecoder decoder(combConfig);
        auto vp = makeNtscVp();
        vp.system = chd::metadata::PAL;
        chd_clear_last_error();
        REQUIRE(!decoder.configure(vp));
        REQUIRE(contains(chd_last_error(), "NTSC"));
    }
    return 0;
}

int testFieldAccessorsAreBounded() {
    chd::metadata::LdDecodeMetaData meta;
    chd::metadata::LdDecodeMetaData::VideoParameters vp = makeNtscVp();
    vp.numberOfSequentialFields = 2;
    meta.setVideoParameters(vp);

    chd::metadata::LdDecodeMetaData::Field field;
    field.isFirstField = true;
    meta.appendField(field);
    field.isFirstField = false;
    meta.appendField(field);
    REQUIRE(meta.getNumberOfFields() == 2);

    // In range: the real rows.
    REQUIRE(meta.getField(1).isFirstField);
    REQUIRE(!meta.getField(2).isFirstField);

    // Out of range on both sides: empty metadata, no read outside the vector.
    // (Under a sanitiser this is the assertion that actually bites.)
    REQUIRE(!meta.getField(0).isFirstField);
    REQUIRE(!meta.getField(-5).isFirstField);
    REQUIRE(!meta.getField(99).isFirstField);
    REQUIRE(meta.getFieldDropOuts(99).size() == 0);
    REQUIRE(meta.getFieldVbi(0).vbiData[0] == 0);

    // Setters out of range leave the real rows alone.
    chd::metadata::LdDecodeMetaData::Field replacement;
    replacement.isFirstField = false;
    meta.updateField(replacement, 99);
    meta.updateField(replacement, 0);
    REQUIRE(meta.getField(1).isFirstField);
    REQUIRE(meta.getNumberOfFields() == 2);

    // A frame number that cannot resolve reports the sentinel rather than
    // looking a field up with it.
    REQUIRE(meta.getFirstFieldNumber(0, true) == -1);
    REQUIRE(meta.getSecondFieldNumber(0, true) == -1);
    return 0;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "chromadec-error-propagation";
    fs::remove_all(dir);
    fs::create_directories(dir);

    int rc = 0;
    if (rc == 0) rc = testFailRecordsDetailWithoutSink();
    if (rc == 0) rc = testVideoOpenStatuses(dir);
    if (rc == 0) rc = testReturnedFailuresAreMarked(dir);
    if (rc == 0) rc = testCombRejectsOversizedGeometry();
    if (rc == 0) rc = testFieldAccessorsAreBounded();

    fs::remove_all(dir);
    if (rc == 0) std::cout << "test_error_propagation: all tests passed\n";
    return rc;
}
