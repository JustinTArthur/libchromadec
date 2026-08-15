// SPDX-License-Identifier: GPL-3.0-or-later

#include "cvbs_metadata_sqlite.h"

#include <sqlite3.h>

#include "../common/error_state.h"
#include "sqlite_query.h"

namespace chd::metadata {

namespace {

const char *kSelectSql =
    "SELECT preset, sample_encoding_preset, signal_state_preset, signal_type, "
    "       decoder, git_branch, git_commit, number_of_sequential_frames, "
    "       black_level, has_nonstandard_values, capture_notes "
    "FROM cvbs_file LIMIT 1";

}  // namespace

std::optional<CvbsMetadata> readCvbsMetadata(const std::string &metaPath)
{
    SqliteDb db;
    if (db.open(metaPath, SQLITE_OPEN_READONLY) != SQLITE_OK) {
        chd::detail::set_last_error("CVBS metadata: could not open " + metaPath + ": " +
                                    db.errmsg());
        return std::nullopt;
    }

    // Confirm the schema's user_version is one the reader implements. The
    // `cvbs_file` columns we select are identical across 7..10; the bumps in
    // that range moved audio metadata around (8 added `audio_locked`, 9
    // replaced it with a per-track table, 10 recast that as
    // `audio_channel_pair` on SMPTE 272M), which this reader does not touch.
    int userVersion = -1;
    {
        sqlite3_stmt *vstmt = nullptr;
        if (sqlite3_prepare_v2(db.handle(), "PRAGMA user_version", -1, &vstmt, nullptr) == SQLITE_OK
            && sqlite3_step(vstmt) == SQLITE_ROW) {
            userVersion = sqlite3_column_int(vstmt, 0);
        }
        if (vstmt != nullptr) sqlite3_finalize(vstmt);
    }
    if (userVersion < 7 || userVersion > 10) {
        chd::detail::set_last_error(
            "CVBS metadata: user_version = " + std::to_string(userVersion) +
            " (expected 7 to 10) in " + metaPath);
        return std::nullopt;
    }

    SqliteQuery query(db);
    if (!query.prepare(kSelectSql) || !query.exec() || !query.next()) {
        chd::detail::set_last_error(
            "CVBS metadata: failed to read cvbs_file row from " + metaPath + ": " +
            query.lastError().text());
        return std::nullopt;
    }

    const std::string presetName    = query.value("preset").toString();
    const std::string encodingName  = query.value("sample_encoding_preset").toString();
    const std::string signalName    = query.value("signal_state_preset").toString();
    const std::string signalTypeStr = query.value("signal_type").toString();

    const chd::format::VideoStandardPreset *standard = chd::format::findVideoStandardByName(presetName);
    if (standard == nullptr) {
        chd::detail::set_last_error("CVBS metadata: unknown video standard preset '" + presetName + "'");
        return std::nullopt;
    }
    const chd::format::SampleEncodingPreset *encoding = chd::format::findSampleEncodingByName(encodingName);
    if (encoding == nullptr) {
        chd::detail::set_last_error("CVBS metadata: unknown sample encoding preset '" + encodingName + "'");
        return std::nullopt;
    }
    const chd::format::SignalStatePreset *state = chd::format::findSignalStateByName(signalName);
    if (state == nullptr) {
        chd::detail::set_last_error("CVBS metadata: unknown signal state preset '" + signalName + "'");
        return std::nullopt;
    }
    CvbsSignalType signalType;
    if (signalTypeStr == "composite") signalType = CvbsSignalType::Composite;
    else if (signalTypeStr == "yc")   signalType = CvbsSignalType::Yc;
    else {
        chd::detail::set_last_error("CVBS metadata: unknown signal_type '" + signalTypeStr + "'");
        return std::nullopt;
    }

    CvbsMetadata meta;
    meta.videoStandard  = standard;
    meta.sampleEncoding = encoding->encoding;
    meta.signalState    = state->state;
    meta.signalType     = signalType;
    meta.decoder        = query.value("decoder").toString();

    auto branchVal = query.value("git_branch");
    if (!branchVal.isNull()) meta.gitBranch = branchVal.toString();
    auto commitVal = query.value("git_commit");
    if (!commitVal.isNull()) meta.gitCommit = commitVal.toString();
    auto framesVal = query.value("number_of_sequential_frames");
    if (!framesVal.isNull()) meta.numberOfSequentialFrames = framesVal.toLongLong();
    auto blackVal = query.value("black_level");
    if (!blackVal.isNull()) meta.blackLevelOverride = blackVal.toInt();
    auto nonstdVal = query.value("has_nonstandard_values");
    if (!nonstdVal.isNull()) meta.hasNonstandardValues = nonstdVal.toBool();
    auto notesVal = query.value("capture_notes");
    if (!notesVal.isNull()) meta.captureNotes = notesVal.toString();

    return meta;
}

}  // namespace chd::metadata