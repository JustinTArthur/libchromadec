/******************************************************************************
 * lddecodemetadata.cpp
 * ld-decode-tools TBC library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2025 Simon Inns
 * SPDX-FileCopyrightText: 2022 Ryan Holtz
 * SPDX-FileCopyrightText: 2022-2023 Adam Sampson
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "core.h"

#include "ld_metadata_sqlite.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <map>
#include <stdexcept>

#include "../common/log.h"

namespace chd::metadata {

// Default values used when configuring VideoParameters for a particular video system.
// See the comments in VideoParameters for the meanings of these values.
// For descriptions of the systems, see ITU BT.1700.
struct VideoSystemDefaults {
    VideoSystem system;
    const char *name;
    double fSC;
    int32_t minActiveFrameLine;
    int32_t firstActiveFrameLine;
    int32_t lastActiveFrameLine;
};

static constexpr VideoSystemDefaults palDefaults {
    PAL,
    "PAL",
    (283.75 * 15625) + 25,
    2,
    // Active frame line range below is inclusive (the last value is the last
    // line that is part of the active picture, not one past it).
    // Interlaced line 44 is PAL line 23 (the first active half-line)
    // Interlaced line 619 is PAL line 623 (the last active half-line)
    44, 619,
};

static constexpr VideoSystemDefaults ntscDefaults {
    NTSC,
    "NTSC",
    315.0e6 / 88.0,
    1,
    // Active frame line range below is inclusive (the last value is the last
    // line that is part of the active picture, not one past it).
    // Interlaced line 39 is the first active half-line
    // Interlaced line 40 is NTSC line 21 (the closed-caption line)
    // Interlaced line 524 is NTSC line 263 (the last active half-line)
    39, 524,
};

static constexpr VideoSystemDefaults palMDefaults {
    PAL_M,
    "PAL_M",
    5.0e6 * (63.0 / 88.0) * (909.0 / 910.0),
    ntscDefaults.minActiveFrameLine,
    ntscDefaults.firstActiveFrameLine, ntscDefaults.lastActiveFrameLine,
};

static constexpr VideoSystemDefaults secamDefaults {
    SECAM,
    "SECAM",
    // SECAM carries FM chroma on two line-alternate subcarriers (fOB = 272*fH,
    // fOR = 282*fH), so no single subcarrier frequency exists; the reference
    // frequency here is the HF pre-correction (anti-bell) centre f0 from
    // BT.1700 Part C Table 4. Raster geometry is shared with PAL.
    4286000.0,
    palDefaults.minActiveFrameLine,
    palDefaults.firstActiveFrameLine, palDefaults.lastActiveFrameLine,
};

// These must be in the same order as enum VideoSystem
static constexpr VideoSystemDefaults VIDEO_SYSTEM_DEFAULTS[] = {
    palDefaults,
    ntscDefaults,
    palMDefaults,
    secamDefaults,
};

// Return appropriate defaults for the selected video system
static const VideoSystemDefaults &getSystemDefaults(const LdDecodeMetaData::VideoParameters &videoParameters)
{
    return VIDEO_SYSTEM_DEFAULTS[videoParameters.system];
}

// Normalise a video-system name for tolerant matching: trim, upper-case, and
// fold '-' and ' ' to '_' so "PAL-M", "pal m" and "PAL_M" all compare equal.
static std::string normaliseVideoSystemName(std::string name)
{
    const auto notSpace = [](unsigned char c) { return std::isspace(c) == 0; };
    name.erase(name.begin(), std::find_if(name.begin(), name.end(), notSpace));
    name.erase(std::find_if(name.rbegin(), name.rend(), notSpace).base(), name.end());

    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) -> char {
        if (c == '-' || c == ' ') return '_';
        return static_cast<char>(std::toupper(c));
    });
    return name;
}

// Look up a video system by name.
// Return true and set system if found; if not found, return false.
bool parseVideoSystemName(std::string name, VideoSystem &system)
{
    const std::string normalisedName = normaliseVideoSystemName(name);

    // Search VIDEO_SYSTEM_DEFAULTS for a matching (normalised) name
    for (const auto &defaults: VIDEO_SYSTEM_DEFAULTS) {
        if (normalisedName == normaliseVideoSystemName(defaults.name)) {
            system = defaults.system;
            return true;
        }
    }

    // Legacy/alternate metadata aliases:
    // MPAL/PALM/PAL_M and Nlinha name the 525-line PAL variant -> PAL-M.
    if (normalisedName == "MPAL" || normalisedName == "PALM" || normalisedName == "PAL_M"
        || normalisedName == "N_LINHA" || normalisedName == "NLINHA"
        || normalisedName == "PAL_N_LINHA" || normalisedName == "PAL_NLINHA") {
        system = PAL_M;
        return true;
    }

    // NTSC aliases used by some tooling/presets.
    if (normalisedName == "NTSCJ" || normalisedName == "NTSC_J") {
        system = NTSC;
        return true;
    }

    // ME-SECAM names the PAL-circuit VHS recording method for SECAM signals;
    // the baseband colour standard is SECAM either way.
    if (normalisedName == "MESECAM" || normalisedName == "ME_SECAM") {
        system = SECAM;
        return true;
    }

    // PAL-family aliases emitted/accepted by vhs-decode workflows, mapped to
    // PAL line-system defaults.
    if (normalisedName == "PALN" || normalisedName == "PAL_N"
        || normalisedName == "405" || normalisedName == "819") {
        system = PAL;
        return true;
    }

    return false;
}

// Read VBI from SQLite
void LdDecodeMetaData::Vbi::read(SqliteReader &reader, int captureId, int fieldId)
{
    int vbi0, vbi1, vbi2;
    if (reader.readFieldVbi(captureId, fieldId, vbi0, vbi1, vbi2)) {
        vbiData[0] = vbi0;
        vbiData[1] = vbi1;
        vbiData[2] = vbi2;
        inUse = true;
    } else {
        inUse = false;
    }
}

// Write VBI to SQLite
void LdDecodeMetaData::Vbi::write(SqliteWriter &writer, int captureId, int fieldId) const
{
    if (inUse) {
        writer.writeFieldVbi(captureId, fieldId, vbiData[0], vbiData[1], vbiData[2]);
    }
}

// Read VideoParameters from SQLite (handled in main read method)
void LdDecodeMetaData::VideoParameters::read(SqliteReader &reader, int captureId)
{
    // This method is no longer used - data is read directly in LdDecodeMetaData::read
}

// Write VideoParameters to SQLite (handled in main write method)
void LdDecodeMetaData::VideoParameters::write(SqliteWriter &writer, int captureId) const
{
    // This method is no longer used - data is written directly in LdDecodeMetaData::write
}

// Read VitsMetrics from SQLite
void LdDecodeMetaData::VitsMetrics::read(SqliteReader &reader, int captureId, int fieldId)
{
    double wSnr, bPsnr;
    if (reader.readFieldVitsMetrics(captureId, fieldId, wSnr, bPsnr)) {
        this->wSNR = wSnr;
        this->bPSNR = bPsnr;
        inUse = true;
    } else {
        inUse = false;
    }
}

// Write VitsMetrics to SQLite
void LdDecodeMetaData::VitsMetrics::write(SqliteWriter &writer, int captureId, int fieldId) const
{
    if (inUse) {
        writer.writeFieldVitsMetrics(captureId, fieldId, wSNR, bPSNR);
    }
}

// Read Ntsc from SQLite (data is read from main field record)
void LdDecodeMetaData::Ntsc::read(SqliteReader &reader, int captureId, int fieldId, ClosedCaption &closedCaption)
{
    // NTSC data is read directly from the field_record table in readFields
    // Closed caption is read separately
    closedCaption.read(reader, captureId, fieldId);
}

// Write Ntsc to SQLite (data is written to main field record)
void LdDecodeMetaData::Ntsc::write(SqliteWriter &writer, int captureId, int fieldId) const
{
    // NTSC data is written directly to the field_record table in Field::write
    // This method is essentially a no-op since the data is handled elsewhere
}

// Read Vitc from SQLite
void LdDecodeMetaData::Vitc::read(SqliteReader &reader, int captureId, int fieldId)
{
    int vitcDataArray[8];
    if (reader.readFieldVitc(captureId, fieldId, vitcDataArray)) {
        for (int i = 0; i < 8; i++) {
            vitcData[i] = vitcDataArray[i];
        }
        inUse = true;
    } else {
        inUse = false;
    }
}

// Write Vitc to SQLite
void LdDecodeMetaData::Vitc::write(SqliteWriter &writer, int captureId, int fieldId) const
{
    if (inUse) {
        int vitcDataArray[8];
        for (int i = 0; i < 8; i++) {
            vitcDataArray[i] = vitcData[i];
        }
        writer.writeFieldVitc(captureId, fieldId, vitcDataArray);
    }
}

// Read ClosedCaption from SQLite
void LdDecodeMetaData::ClosedCaption::read(SqliteReader &reader, int captureId, int fieldId)
{
    int data0Val, data1Val;
    if (reader.readFieldClosedCaption(captureId, fieldId, data0Val, data1Val)) {
        data0 = data0Val;
        data1 = data1Val;
        inUse = true;
    } else {
        inUse = false;
    }
}

// Write ClosedCaption to SQLite
void LdDecodeMetaData::ClosedCaption::write(SqliteWriter &writer, int captureId, int fieldId) const
{
    if (inUse) {
        writer.writeFieldClosedCaption(captureId, fieldId, data0, data1);
    }
}

// Read PcmAudioParameters from SQLite (handled in main read method)
void LdDecodeMetaData::PcmAudioParameters::read(SqliteReader &reader, int captureId)
{
    // This method is no longer used - data is read directly in LdDecodeMetaData::read
}

// Write PcmAudioParameters to SQLite (handled in main write method)
void LdDecodeMetaData::PcmAudioParameters::write(SqliteWriter &writer, int captureId) const
{
    // This method is no longer used - data is written directly in LdDecodeMetaData::write
}

// Read Field from SQLite (data is read in readFields)
void LdDecodeMetaData::Field::read(SqliteReader &reader, int captureId)
{
    // This method is no longer used - data is read directly in LdDecodeMetaData::readFields
}

// Write Field to SQLite
void LdDecodeMetaData::Field::write(SqliteWriter &writer, int captureId) const
{
    // Convert seqNo (1-indexed) to fieldId (0-indexed)
    int fieldId = seqNo - 1;
    
    // Write main field record with NTSC data embedded
    writer.writeField(captureId, fieldId, audioSamples, decodeFaults, diskLoc,
                     efmTValues, fieldPhaseID, fileLoc, isFirstField, medianBurstIRE,
                     pad, syncConf, ntsc.isFmCodeDataValid, ntsc.fmCodeData,
                     ntsc.fieldFlag, ntsc.isVideoIdDataValid, ntsc.videoIdData,
                     ntsc.whiteFlag);

    // Write optional field data
    vitsMetrics.write(writer, captureId, fieldId);
    vbi.write(writer, captureId, fieldId);
    vitc.write(writer, captureId, fieldId);
    closedCaption.write(writer, captureId, fieldId);
    dropOuts.write(writer, captureId, fieldId);
}

LdDecodeMetaData::LdDecodeMetaData()
{
    clear();
}

// Reset the metadata to the defaults
void LdDecodeMetaData::clear()
{
    // Default to the standard still-frame field order (of first field first)
    isFirstFieldFirst = true;

    // Reset the parameters to their defaults
    videoParameters = VideoParameters();
    pcmAudioParameters = PcmAudioParameters();

    fields.clear();
}

// Dispatch metadata reading by file extension. JSON sidecars (`.tbc.json`,
// pre-2025 ld-decode format) route to readJsonImpl; everything else is
// treated as a SQLite sidecar (`.tbc.db`, current format).
bool LdDecodeMetaData::read(std::string fileName)
{
    auto endsWith = [](const std::string &s, const std::string &suffix) {
        return s.size() >= suffix.size()
            && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                          [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    };

    if (endsWith(fileName, ".json")) {
        return readJsonImpl(fileName);
    }
    return readSqliteImpl(fileName);
}

// Read all metadata from SQLite file
bool LdDecodeMetaData::readSqliteImpl(const std::string &fileName)
{
    if (!std::filesystem::exists(fileName)) {
        chd::log::fail() << "SQLite input file does not exist:" << fileName;
        return false;
    }

    clear();

    try {
        SqliteReader reader(fileName);
        
        int captureId;
        std::string system, decoder, gitBranch, gitCommit, captureNotes;
        double videoSampleRate;
        int activeVideoStart, activeVideoEnd, fieldWidth, fieldHeight, numberOfSequentialFields;
        int colourBurstStart, colourBurstEnd, white16bIre, black16bIre, blanking16bIre;
        bool isMapped, isSubcarrierLocked, isWidescreen;
        int sidecarFirstActiveFrameLine = -1, sidecarLastActiveFrameLine = -1;

        // Read capture metadata
        if (!reader.readCaptureMetadata(captureId, system, decoder, gitBranch, gitCommit,
                                       videoSampleRate, activeVideoStart, activeVideoEnd,
                                       fieldWidth, fieldHeight, numberOfSequentialFields,
                                       colourBurstStart, colourBurstEnd, isMapped,
                                       isSubcarrierLocked, isWidescreen, white16bIre,
                                       black16bIre, blanking16bIre, captureNotes,
                                       sidecarFirstActiveFrameLine, sidecarLastActiveFrameLine)) {
            chd::log::fail() << "Failed to read capture metadata from SQLite file";
            return false;
        }

        // Set video parameters
        videoParameters.numberOfSequentialFields = numberOfSequentialFields;
        if (!parseVideoSystemName(system, videoParameters.system)) {
            chd::log::fail() << "Unknown video system:" << system;
            return false;
        }
        videoParameters.isSubcarrierLocked = isSubcarrierLocked;
        videoParameters.isWidescreen = isWidescreen;
        videoParameters.colourBurstStart = colourBurstStart;
        videoParameters.colourBurstEnd = colourBurstEnd;
        videoParameters.activeVideoStart = activeVideoStart;
        videoParameters.activeVideoEnd = activeVideoEnd;
        videoParameters.white16bIre = white16bIre;
        videoParameters.black16bIre = black16bIre;
        videoParameters.blanking16bIre = blanking16bIre;
        videoParameters.fieldWidth = fieldWidth;
        videoParameters.fieldHeight = fieldHeight;
        videoParameters.sampleRate = videoSampleRate;
        videoParameters.isMapped = isMapped;
        videoParameters.tapeFormat = captureNotes;
        videoParameters.gitBranch = gitBranch;
        videoParameters.gitCommit = gitCommit;
        videoParameters.isValid = true;

        // Read PCM audio parameters if they exist
        int bits;
        bool isSigned, isLittleEndian;
        double audioSampleRate;
        if (reader.readPcmAudioParameters(captureId, bits, isSigned, isLittleEndian, audioSampleRate)) {
            pcmAudioParameters.bits = bits;
            pcmAudioParameters.isSigned = isSigned;
            pcmAudioParameters.isLittleEndian = isLittleEndian;
            pcmAudioParameters.sampleRate = audioSampleRate;
            pcmAudioParameters.isValid = true;
        }

        // Read all fields
        readFields(reader, captureId);

        // Now we know the video system, initialise the rest of VideoParameters.
        // Run inside the try block so the sidecar's active-line overrides are
        // applied before we leave (-1s here = no override; standard defaults stay).
        initialiseVideoSystemParameters();

        if (sidecarFirstActiveFrameLine >= 0 || sidecarLastActiveFrameLine >= 0) {
            LineParameters sidecarLines;
            sidecarLines.firstActiveFrameLine = sidecarFirstActiveFrameLine;
            sidecarLines.lastActiveFrameLine  = sidecarLastActiveFrameLine;
            processLineParameters(sidecarLines);
        }

    } catch (SqliteReader::Error &error) {
        chd::log::fail() << "Reading SQLite file failed:" << error.what();
        return false;
    }

    // Generate the PCM audio map based on the field metadata
    generatePcmAudioMap();

    return true;
}

// Write all metadata out to an SQLite file
bool LdDecodeMetaData::write(std::string fileName) const
{
    // Check if we're updating an existing file or creating a new one
    bool isUpdate = std::filesystem::exists(fileName);
    int captureId = 1; // Default for new files
    
    if (isUpdate) {
        // Try to read the existing capture_id from the file
        try {
            SqliteReader reader(fileName);
            std::string existingSystem, existingDecoder, existingGitBranch, existingGitCommit, existingCaptureNotes;
            double existingVideoSampleRate;
            int existingActiveVideoStart, existingActiveVideoEnd, existingFieldWidth, existingFieldHeight;
            int existingNumberOfSequentialFields, existingColourBurstStart, existingColourBurstEnd;
            int existingWhite16bIre, existingBlack16bIre, existingBlanking16bIre;
            bool existingIsMapped, existingIsSubcarrierLocked, existingIsWidescreen;
            int existingFirstActiveFrameLine, existingLastActiveFrameLine;

            if (reader.readCaptureMetadata(captureId, existingSystem, existingDecoder,
                                         existingGitBranch, existingGitCommit, existingVideoSampleRate,
                                         existingActiveVideoStart, existingActiveVideoEnd,
                                         existingFieldWidth, existingFieldHeight, existingNumberOfSequentialFields,
                                         existingColourBurstStart, existingColourBurstEnd,
                                         existingIsMapped, existingIsSubcarrierLocked, existingIsWidescreen,
                                         existingWhite16bIre, existingBlack16bIre, existingBlanking16bIre, existingCaptureNotes,
                                         existingFirstActiveFrameLine, existingLastActiveFrameLine)) {
                chd::log::debug() << "Updating existing SQLite file with capture_id:" << captureId;
            } else {
                chd::log::warn() << "Could not read existing capture metadata, treating as new file";
                isUpdate = false;
                captureId = 1;
            }
        } catch (...) {
            chd::log::warn() << "Error reading existing SQLite file, treating as new file";
            isUpdate = false;
            captureId = 1;
        }
    }

    try {
        SqliteWriter writer(fileName);
        
        // Only create schema for new files
        if (!isUpdate) {
            if (!writer.createSchema()) {
                chd::log::fail() << "Failed to create SQLite schema";
                return false;
            }
        }

        if (!writer.beginTransaction()) {
            chd::log::fail() << "Failed to begin transaction";
            return false;
        }

        // Write or update capture metadata
        std::string systemName = getVideoSystemDescription();
        if (isUpdate) {
            // Update existing capture metadata
            if (!writer.updateCaptureMetadata(captureId, systemName, "ld-decode", // TODO: make decoder configurable
                                            videoParameters.gitBranch, videoParameters.gitCommit,
                                            videoParameters.sampleRate, videoParameters.activeVideoStart, 
                                            videoParameters.activeVideoEnd, videoParameters.fieldWidth,
                                            videoParameters.fieldHeight, videoParameters.numberOfSequentialFields,
                                            videoParameters.colourBurstStart, videoParameters.colourBurstEnd,
                                            videoParameters.isMapped, videoParameters.isSubcarrierLocked,
                                            videoParameters.isWidescreen, videoParameters.white16bIre,
                                            videoParameters.black16bIre, videoParameters.blanking16bIre, videoParameters.tapeFormat)) {
                writer.rollbackTransaction();
                return false;
            }
        } else {
            // Create new capture metadata
            captureId = writer.writeCaptureMetadata(
                systemName, "ld-decode", // TODO: make decoder configurable
                videoParameters.gitBranch, videoParameters.gitCommit,
                videoParameters.sampleRate, videoParameters.activeVideoStart, 
                videoParameters.activeVideoEnd, videoParameters.fieldWidth,
                videoParameters.fieldHeight, videoParameters.numberOfSequentialFields,
                videoParameters.colourBurstStart, videoParameters.colourBurstEnd,
                videoParameters.isMapped, videoParameters.isSubcarrierLocked,
                videoParameters.isWidescreen, videoParameters.white16bIre,
                videoParameters.black16bIre, videoParameters.blanking16bIre, videoParameters.tapeFormat);

            if (captureId == -1) {
                writer.rollbackTransaction();
                return false;
            }
        }

        // Write PCM audio parameters if they exist
        if (pcmAudioParameters.isValid) {
            if (!writer.writePcmAudioParameters(captureId, pcmAudioParameters.bits,
                                              pcmAudioParameters.isSigned,
                                              pcmAudioParameters.isLittleEndian,
                                              pcmAudioParameters.sampleRate)) {
                writer.rollbackTransaction();
                return false;
            }
        }

        // Write all fields
        writeFields(writer, captureId);

        if (!writer.commitTransaction()) {
            chd::log::fail() << "Failed to commit transaction";
            return false;
        }

    } catch (SqliteWriter::Error &error) {
        chd::log::fail() << "Writing SQLite file failed:" << error.what();
        return false;
    }

    return true;
}

// Read array of Fields from SQLite (optimized version)
void LdDecodeMetaData::readFields(SqliteReader &reader, int captureId)
{
    SqliteQuery fieldsQuery;
    if (!reader.readFields(captureId, fieldsQuery)) {
        return;
    }

    // Pre-read all optional field data in bulk for performance
    SqliteQuery vitsQuery, vbiQuery, vitcQuery, ccQuery, dropoutsQuery;
    reader.readAllFieldVitsMetrics(captureId, vitsQuery);
    reader.readAllFieldVbi(captureId, vbiQuery);
    reader.readAllFieldVitc(captureId, vitcQuery);
    reader.readAllFieldClosedCaptions(captureId, ccQuery);
    reader.readAllFieldDropouts(captureId, dropoutsQuery);

    // Create lookup maps for fast field data retrieval
    std::map<int, std::pair<double, double>> vitsMap;
    std::map<int, std::vector<int>> vbiMap, vitcMap, ccMap;
    std::multimap<int, std::vector<int>> dropoutsMap;

    // Populate VITS metrics map
    while (vitsQuery.next()) {
        int fieldId = vitsQuery.value("field_id").toInt();
        double wSnr = vitsQuery.value("w_snr").toDouble();
        double bPsnr = vitsQuery.value("b_psnr").toDouble();
        vitsMap[fieldId] = std::make_pair(wSnr, bPsnr);
    }

    // Populate VBI map
    while (vbiQuery.next()) {
        int fieldId = vbiQuery.value("field_id").toInt();
        std::vector<int> vbiData = {vbiQuery.value("vbi0").toInt(), 
                               vbiQuery.value("vbi1").toInt(), 
                               vbiQuery.value("vbi2").toInt()};
        vbiMap[fieldId] = vbiData;
    }

    // Populate VITC map
    while (vitcQuery.next()) {
        int fieldId = vitcQuery.value("field_id").toInt();
        std::vector<int> vitcData;
        for (int i = 0; i < 8; i++) {
            vitcData.push_back(vitcQuery.value(("vitc" + std::to_string(i)).c_str()).toInt());
        }
        vitcMap[fieldId] = vitcData;
    }

    // Populate closed captions map
    while (ccQuery.next()) {
        int fieldId = ccQuery.value("field_id").toInt();
        std::vector<int> ccData = {ccQuery.value("data0").toInt(), 
                              ccQuery.value("data1").toInt()};
        ccMap[fieldId] = ccData;
    }

    // Populate dropouts map
    while (dropoutsQuery.next()) {
        int fieldId = dropoutsQuery.value("field_id").toInt();
        std::vector<int> dropoutData = {dropoutsQuery.value("startx").toInt(),
                                   dropoutsQuery.value("endx").toInt(),
                                   dropoutsQuery.value("field_line").toInt()};
        dropoutsMap.emplace(fieldId, dropoutData);
    }

    // Process main field records and apply cached data
    while (fieldsQuery.next()) {
        Field field;
        
        // Note: field_id in database is 0-indexed, but seqNo should be 1-indexed
        int fieldId = fieldsQuery.value("field_id").toInt();
        field.seqNo = fieldId + 1;
        field.isFirstField = SqliteValue::toBoolOrDefault(fieldsQuery, "is_first_field");
        field.syncConf = SqliteValue::toIntOrDefault(fieldsQuery, "sync_conf", 0);
        field.medianBurstIRE = SqliteValue::toDoubleOrDefault(fieldsQuery, "median_burst_ire", 0.0);
        field.fieldPhaseID = SqliteValue::toIntOrDefault(fieldsQuery, "field_phase_id");
        field.audioSamples = SqliteValue::toIntOrDefault(fieldsQuery, "audio_samples");
        field.diskLoc = SqliteValue::toDoubleOrDefault(fieldsQuery, "disk_loc");
        field.fileLoc = SqliteValue::toLongLongOrDefault(fieldsQuery, "file_loc");
        field.decodeFaults = SqliteValue::toIntOrDefault(fieldsQuery, "decode_faults");
        field.efmTValues = SqliteValue::toIntOrDefault(fieldsQuery, "efm_t_values");
        field.pad = SqliteValue::toBoolOrDefault(fieldsQuery, "pad");

        // Read NTSC data from the main field record
        field.ntsc.isFmCodeDataValid = fieldsQuery.value("ntsc_is_fm_code_data_valid").toInt() == 1;
        field.ntsc.fmCodeData = fieldsQuery.value("ntsc_fm_code_data").toInt();
        field.ntsc.fieldFlag = fieldsQuery.value("ntsc_field_flag").toInt() == 1;
        field.ntsc.isVideoIdDataValid = fieldsQuery.value("ntsc_is_video_id_data_valid").toInt() == 1;
        field.ntsc.videoIdData = fieldsQuery.value("ntsc_video_id_data").toInt();
        field.ntsc.whiteFlag = fieldsQuery.value("ntsc_white_flag").toInt() == 1;
        field.ntsc.inUse = field.ntsc.isFmCodeDataValid || field.ntsc.isVideoIdDataValid;

        // Apply cached optional field data
        if (vitsMap.count(fieldId) > 0) {
            field.vitsMetrics.wSNR = vitsMap[fieldId].first;
            field.vitsMetrics.bPSNR = vitsMap[fieldId].second;
            field.vitsMetrics.inUse = true;
        }

        if (vbiMap.count(fieldId) > 0) {
            std::vector<int> vbiData = vbiMap[fieldId];
            field.vbi.vbiData[0] = vbiData[0];
            field.vbi.vbiData[1] = vbiData[1]; 
            field.vbi.vbiData[2] = vbiData[2];
            field.vbi.inUse = true;
        }

        if (vitcMap.count(fieldId) > 0) {
            std::vector<int> vitcData = vitcMap[fieldId];
            for (int i = 0; i < 8 && i < vitcData.size(); i++) {
                field.vitc.vitcData[i] = vitcData[i];
            }
            field.vitc.inUse = true;
        }

        if (ccMap.count(fieldId) > 0) {
            std::vector<int> ccData = ccMap[fieldId];
            field.closedCaption.data0 = ccData[0];
            field.closedCaption.data1 = ccData[1];
            field.closedCaption.inUse = true;
        }

        if (dropoutsMap.count(fieldId) > 0) {
            field.dropOuts.clear();
            auto range = dropoutsMap.equal_range(fieldId);
            for (auto it = range.first; it != range.second; ++it) {
                const auto &dropout = it->second;
                field.dropOuts.append(dropout[0], dropout[1], dropout[2]);
            }
        }

        fields.push_back(field);
    }
}

// Write array of Fields to SQLite
void LdDecodeMetaData::writeFields(SqliteWriter &writer, int captureId) const
{
    for (const Field &field : fields) {
        field.write(writer, captureId);
    }
}

// This method returns the videoParameters metadata
const LdDecodeMetaData::VideoParameters &LdDecodeMetaData::getVideoParameters()
{
    if (!videoParameters.isValid) {
        throw std::runtime_error("VideoParameters not initialized - metadata file may not have been read successfully");
    }
    return videoParameters;
}

// This method sets the videoParameters metadata
void LdDecodeMetaData::setVideoParameters(const LdDecodeMetaData::VideoParameters &_videoParameters)
{
    videoParameters = _videoParameters;
    videoParameters.isValid = true;
}

// This method returns the pcmAudioParameters metadata
const LdDecodeMetaData::PcmAudioParameters &LdDecodeMetaData::getPcmAudioParameters()
{
    if (!pcmAudioParameters.isValid) {
        throw std::runtime_error("PcmAudioParameters not initialized - metadata file may not have been read successfully");
    }
    return pcmAudioParameters;
}

// This method sets the pcmAudioParameters metadata
void LdDecodeMetaData::setPcmAudioParameters(const LdDecodeMetaData::PcmAudioParameters &_pcmAudioParameters)
{
    pcmAudioParameters = _pcmAudioParameters;
    pcmAudioParameters.isValid = true;
}

// Based on the video system selected, set default values for the members of
// VideoParameters that aren't obtained from the metadata.
void LdDecodeMetaData::initialiseVideoSystemParameters()
{
    const VideoSystemDefaults &defaults = getSystemDefaults(videoParameters);
    videoParameters.fSC = defaults.fSC;

    // Set default LineParameters
    LdDecodeMetaData::LineParameters lineParameters;
    processLineParameters(lineParameters);
}

// Validate LineParameters and apply them to the VideoParameters
void LdDecodeMetaData::processLineParameters(LdDecodeMetaData::LineParameters &lineParameters)
{
    lineParameters.applyTo(videoParameters);
}

// Re-declare the video system, rederiving system defaults and revalidating
// the active-line ranges against the new system's bounds
void LdDecodeMetaData::overrideVideoSystem(VideoSystem system)
{
    videoParameters.system = system;
    videoParameters.fSC = getSystemDefaults(videoParameters).fSC;

    LineParameters lines;
    lines.firstActiveFrameLine = videoParameters.firstActiveFrameLine;
    lines.lastActiveFrameLine  = videoParameters.lastActiveFrameLine;
    processLineParameters(lines);
}

// Validate and apply to a set of VideoParameters
void LdDecodeMetaData::LineParameters::applyTo(LdDecodeMetaData::VideoParameters &videoParameters)
{
    const bool firstFrameLineExists = firstActiveFrameLine != -1;
    const bool lastFrameLineExists = lastActiveFrameLine != -1;

    const VideoSystemDefaults &defaults = getSystemDefaults(videoParameters);
    const int32_t minFirstFrameLine = defaults.minActiveFrameLine;
    const int32_t defaultFirstFrameLine = defaults.firstActiveFrameLine;
    const int32_t defaultLastFrameLine = defaults.lastActiveFrameLine;

    // Validate and potentially fix the first active frame line.
    if (firstActiveFrameLine < minFirstFrameLine || firstActiveFrameLine > defaultLastFrameLine) {
        if (firstFrameLineExists) {
            chd::log::warn().nospace() << "Specified first active frame line " << firstActiveFrameLine << " out of bounds (" << minFirstFrameLine << " to "
                              << defaultLastFrameLine << "), resetting to default (" << defaultFirstFrameLine << ").";
        }
        firstActiveFrameLine = defaultFirstFrameLine;
    }

    // Validate and potentially fix the last active frame line.
    if (lastActiveFrameLine < minFirstFrameLine || lastActiveFrameLine > defaultLastFrameLine) {
        if (lastFrameLineExists) {
            chd::log::warn().nospace() << "Specified last active frame line " << lastActiveFrameLine << " out of bounds (" << minFirstFrameLine << " to "
                              << defaultLastFrameLine << "), resetting to default (" << defaultLastFrameLine << ").";
        }
        lastActiveFrameLine = defaultLastFrameLine;
    }

    // Range-check the first and last active frame lines.
    if (firstActiveFrameLine > lastActiveFrameLine) {
        chd::log::warn().nospace() << "Specified last active frame line " << lastActiveFrameLine << " is before specified first active frame line "
                          << firstActiveFrameLine << ", resetting to defaults (" << defaultFirstFrameLine << "-" << defaultLastFrameLine << ").";
        firstActiveFrameLine = defaultFirstFrameLine;
        lastActiveFrameLine = defaultLastFrameLine;
    }

    // Store the new values back into videoParameters
    videoParameters.firstActiveFrameLine = firstActiveFrameLine;
    videoParameters.lastActiveFrameLine = lastActiveFrameLine;
}

// Stand-in for an out-of-range field request. The accessors below used to log
// the out-of-range index and then index with it anyway, which reads outside the
// vector; they return a reference, so there is no status to hand back and this
// is what they yield instead. Callers that cannot tolerate empty metadata
// range-check the field number themselves (see SourceField::loadFields).
static const LdDecodeMetaData::Field &emptyField()
{
    static const LdDecodeMetaData::Field empty{};
    return empty;
}

// This method gets the metadata for the specified sequential field number (indexed from 1 (not 0!))
const LdDecodeMetaData::Field &LdDecodeMetaData::getField(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::getField(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return emptyField();
    }

    return fields[fieldNumber];
}

// This method gets the VITS metrics metadata for the specified sequential field number
const LdDecodeMetaData::VitsMetrics &LdDecodeMetaData::getFieldVitsMetrics(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::getFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return emptyField().vitsMetrics;
    }

    return fields[fieldNumber].vitsMetrics;
}

// This method gets the VBI metadata for the specified sequential field number
const LdDecodeMetaData::Vbi &LdDecodeMetaData::getFieldVbi(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::getFieldVbi(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return emptyField().vbi;
    }

    return fields[fieldNumber].vbi;
}

// This method gets the NTSC metadata for the specified sequential field number
const LdDecodeMetaData::Ntsc &LdDecodeMetaData::getFieldNtsc(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::getFieldNtsc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return emptyField().ntsc;
    }

    return fields[fieldNumber].ntsc;
}

// This method gets the VITC metadata for the specified sequential field number
const LdDecodeMetaData::Vitc &LdDecodeMetaData::getFieldVitc(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::getFieldVitc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return emptyField().vitc;
    }

    return fields[fieldNumber].vitc;
}

// This method gets the Closed Caption metadata for the specified sequential field number
const LdDecodeMetaData::ClosedCaption &LdDecodeMetaData::getFieldClosedCaption(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::getFieldClosedCaption(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return emptyField().closedCaption;
    }

    return fields[fieldNumber].closedCaption;
}

// This method gets the drop-out metadata for the specified sequential field number
const DropOuts &LdDecodeMetaData::getFieldDropOuts(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::getFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return emptyField().dropOuts;
    }

    return fields[fieldNumber].dropOuts;
}

// This method sets the field metadata for a field
void LdDecodeMetaData::updateField(const LdDecodeMetaData::Field &field, int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::updateField(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber] = field;
}

// This method sets the field VBI metadata for a field
void LdDecodeMetaData::updateFieldVitsMetrics(const LdDecodeMetaData::VitsMetrics &vitsMetrics, int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::updateFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber].vitsMetrics = vitsMetrics;
}

// This method sets the field VBI metadata for a field
void LdDecodeMetaData::updateFieldVbi(const LdDecodeMetaData::Vbi &vbi, int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::updateFieldVbi(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber].vbi = vbi;
}

// This method sets the field NTSC metadata for a field
void LdDecodeMetaData::updateFieldNtsc(const LdDecodeMetaData::Ntsc &ntsc, int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::updateFieldNtsc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber].ntsc = ntsc;
}

// This method sets the VITC metadata for a field
void LdDecodeMetaData::updateFieldVitc(const LdDecodeMetaData::Vitc &vitc, int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::updateFieldVitc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber].vitc = vitc;
}

// This method sets the Closed Caption metadata for a field
void LdDecodeMetaData::updateFieldClosedCaption(const LdDecodeMetaData::ClosedCaption &closedCaption, int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::updateFieldClosedCaption(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber].closedCaption = closedCaption;
}

// This method sets the field dropout metadata for a field
void LdDecodeMetaData::updateFieldDropOuts(const DropOuts &dropOuts, int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::updateFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber].dropOuts = dropOuts;
}

// This method clears the field dropout metadata for a field
void LdDecodeMetaData::clearFieldDropOuts(int32_t sequentialFieldNumber)
{
    int32_t fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        chd::log::error() << "LdDecodeMetaData::clearFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
        return;
    }

    fields[fieldNumber].dropOuts.clear();
}

// This method appends a new field to the existing metadata
void LdDecodeMetaData::appendField(const LdDecodeMetaData::Field &field)
{
    // Ensure sequential numbering stays contiguous when writing out
    LdDecodeMetaData::Field fieldCopy = field;
    fieldCopy.seqNo = fields.size() + 1;
    fields.push_back(fieldCopy);

    videoParameters.numberOfSequentialFields = fields.size();
}

// Method to get the available number of fields (according to the metadata)
int32_t LdDecodeMetaData::getNumberOfFields()
{
    return fields.size();
}

// Method to set the available number of fields
// XXX This is unnecessary given appendField
void LdDecodeMetaData::setNumberOfFields(int32_t numberOfFields)
{
    videoParameters.numberOfSequentialFields = numberOfFields;
}

// A note about fields, frames and still-frames:
//
// There is a lot of confusing terminology around fields and the order in which
// they should be combined to make a 'frame'.  Basically, (taking NTSC as an example)
// a frame consists of frame lines numbered from 1 to 525.  A frame is made from two
// fields, one field contains field lines 1 to 262.5 and another 263 to 525 (although
// for convenence the 'half-lines' are usually treated as one full line and ignored
// so both fields contain a total of 263 lines of which 1 is ignored).
//
// When a frame is created, the field containing field lines 1-263 is interlaced
// with the field containing 263-525 creating a frame with field lines 1, 263, 2, 264
// and so on.  This 'frame' is then considered to contain frame lines 1-525.
//
// The field containing the first line of the frame is called the 'first field' and
// the field containing the second line of the frame is called the 'second field'.
//
// However, other names exist:
//
// Even/Odd - where the 'odd' field contains the odd line numbers 1, 3, 5, etc. This is
// the same as the 'first' field so odd = first and even = second.
//
// Upper/lower - where the 'upper' field contains the upper-part of each combination.
// This is the same as the first field so upper = first and lower = second.
//
// With a standard TV, as long as one field is first and the other is second, the only
// thing a TV requires is that the sequence of fields is constant.  They are simply
// displayed one set of fields after another to form a frame which is part of a
// moving image.
//
// This is an issue for 'still-frames' as, if the video sequence consists of
// still images (rather than motion), pausing at any given point can result in a
// frame containing a first field from one image and a second field from another
// as there is no concept of 'frame' in the video (just a sequence of first and
// second fields).
//
// Since digital formats are frame based (not field) this is an issue, as there
// is no way (from the video data) to tell how to combine fields into a
// still-frame (rather than just 'a frame').  The LaserDisc mastering could be
// in the order first field/second field = still-frame or second field/first field
// = still frame.
//
// This is why the following methods use the "isFirstFieldFirst" flag (which is
// a little confusing in itself).
//
// There are two ways to determine the 'isFirstFieldFirst'.  The first method is by
// user observation (it's pretty clear on a still-frame when it is wrong), the other
// (used by LaserDisc players) is to look for a CAV picture number in the VBI data
// of the field.  The IEC specification states that the picture number should only be
// in the first field of a frame. (Note: CLV discs don't really have to follow this
// as there are no 'still-frames' allowed by the original format).
//
// This gets even more confusing for NTSC discs using pull-down, where the sequence
// of fields making up the frames isn't even, so some field-pairs aren't considered
// to contain the first field of a still-frame (and when pausing the LaserDisc
// player will never use certain fields to render the still-frame).  Wikipedia is
// your friend if you want to learn more about it.
//
// Determining the correct setting of 'isFirstFieldFirst' is therefore outside of
// the shared-library scope.

// Method to get the available number of still-frames
int32_t LdDecodeMetaData::getNumberOfFrames()
{
    int32_t frameOffset = 0;

    // If the first field in the TBC input isn't the expected first field,
    // skip it when counting the number of still-frames
    if (isFirstFieldFirst) {
        // Expecting first field first
        if (!getField(1).isFirstField) frameOffset = 1;
    } else {
        // Expecting second field first
        if (getField(1).isFirstField) frameOffset = 1;
    }

    return (getNumberOfFields() / 2) - frameOffset;
}

// Method to get the first and second field numbers based on the frame number
// If field = 1 return the firstField, otherwise return second field
// Every caller of the -1 sentinel recovers from it rather than propagating a
// failure: SourceField::loadFields substitutes blank frames and warns, the
// extra-source and VBI scans skip the frame. So these are error() rather than
// fail() - nobody is going to be handed a status carrying the reason, and
// recording it as the thread's error detail would leave a message behind for a
// call that goes on to succeed.
int32_t LdDecodeMetaData::getFieldNumber(int32_t frameNumber, int32_t field)
{
    int32_t firstFieldNumber = 0;
    int32_t secondFieldNumber = 0;

    // Verify the frame number
    if (frameNumber < 1) {
        chd::log::error() << "Invalid frame number, cannot determine fields";
        return -1;
    }

    // Calculate the first and last fields based on the position in the TBC
    if (isFirstFieldFirst) {
        // Expecting TBC file to provide still-frames as first field / second field
        firstFieldNumber = (frameNumber * 2) - 1;
        secondFieldNumber = firstFieldNumber + 1;
    } else {
        // Expecting TBC file to provide still-frames as second field / first field
        secondFieldNumber = (frameNumber * 2) - 1;
        firstFieldNumber = secondFieldNumber + 1;
    }

    // If the field number pointed to by firstFieldNumber doesn't have
    // isFirstField set, move forward field by field until the current
    // field does
    while (!getField(firstFieldNumber).isFirstField) {
        firstFieldNumber++;
        secondFieldNumber++;

        // Give up if we reach the end of the available fields
        if (firstFieldNumber > getNumberOfFields() || secondFieldNumber > getNumberOfFields()) {
            chd::log::error() << "Attempting to get field number failed - no isFirstField in metadata before end of file";
            firstFieldNumber = -1;
            secondFieldNumber = -1;
            break;
        }
    }

    // Range check the first field number
    if (firstFieldNumber > getNumberOfFields()) {
        chd::log::error() << "first field number exceeds the available number of fields";
        firstFieldNumber = -1;
        secondFieldNumber = -1;
    }

    // Range check the second field number
    if (secondFieldNumber > getNumberOfFields()) {
        chd::log::error() << "second field number exceeds the available number of fields";
        firstFieldNumber = -1;
        secondFieldNumber = -1;
    }

    // Both numbers were invalidated above; the caller gets the sentinel rather
    // than a field lookup on it.
    if (firstFieldNumber == -1) return -1;

    // Test for a buggy TBC file...
    if (getField(secondFieldNumber).isFirstField) {
        chd::log::warn() << "LdDecodeMetaData::getFieldNumber(): Both of the determined fields have isFirstField set - the TBC source video is probably broken...";
    }

    if (field == 1) return firstFieldNumber; else return secondFieldNumber;
}

// Method to get the first field number based on the frame number
int32_t LdDecodeMetaData::getFirstFieldNumber(int32_t frameNumber)
{
    return getFieldNumber(frameNumber, 1);
}

// Method to get the second field number based on the frame number
int32_t LdDecodeMetaData::getSecondFieldNumber(int32_t frameNumber)
{
    return getFieldNumber(frameNumber, 2);
}

// Method to set the isFirstFieldFirst flag
void LdDecodeMetaData::setIsFirstFieldFirst(bool flag)
{
    isFirstFieldFirst = flag;
}

// Method to get the isFirstFieldFirst flag
bool LdDecodeMetaData::getIsFirstFieldFirst()
{
    return isFirstFieldFirst;
}

// Method to convert a CLV time code into an equivalent frame number (to make
// processing the timecodes easier)
int32_t LdDecodeMetaData::convertClvTimecodeToFrameNumber(LdDecodeMetaData::ClvTimecode clvTimeCode)
{
    // Calculate the frame number
    int32_t frameNumber = 0;
    VideoParameters videoParameters = getVideoParameters();

    // Check for invalid CLV timecode
    if (clvTimeCode.hours == -1 || clvTimeCode.minutes == -1 || clvTimeCode.seconds == -1 || clvTimeCode.pictureNumber == -1) {
        return -1;
    }

    if (clvTimeCode.hours != -1) {
        if (videoParameters.system == PAL) frameNumber += clvTimeCode.hours * 3600 * 25;
        else frameNumber += clvTimeCode.hours * 3600 * 30;
    }

    if (clvTimeCode.minutes != -1) {
        if (videoParameters.system == PAL) frameNumber += clvTimeCode.minutes * 60 * 25;
        else frameNumber += clvTimeCode.minutes * 60 * 30;
    }

    if (clvTimeCode.seconds != -1) {
        if (videoParameters.system == PAL) frameNumber += clvTimeCode.seconds * 25;
        else frameNumber += clvTimeCode.seconds * 30;
    }

    if (clvTimeCode.pictureNumber != -1) {
        frameNumber += clvTimeCode.pictureNumber;
    }

    return frameNumber;
}

// Method to convert a frame number into an equivalent CLV timecode
LdDecodeMetaData::ClvTimecode LdDecodeMetaData::convertFrameNumberToClvTimecode(int32_t frameNumber)
{
    ClvTimecode clvTimecode;

    clvTimecode.hours = 0;
    clvTimecode.minutes = 0;
    clvTimecode.seconds = 0;
    clvTimecode.pictureNumber = 0;

    if (getVideoParameters().system == PAL) {
        clvTimecode.hours = frameNumber / (3600 * 25);
        frameNumber -= clvTimecode.hours * (3600 * 25);

        clvTimecode.minutes = frameNumber / (60 * 25);
        frameNumber -= clvTimecode.minutes * (60 * 25);

        clvTimecode.seconds = frameNumber / 25;
        frameNumber -= clvTimecode.seconds * 25;

        clvTimecode.pictureNumber = frameNumber;
    } else {
        clvTimecode.hours = frameNumber / (3600 * 30);
        frameNumber -= clvTimecode.hours * (3600 * 30);

        clvTimecode.minutes = frameNumber / (60 * 30);
        frameNumber -= clvTimecode.minutes * (60 * 30);

        clvTimecode.seconds = frameNumber / 30;
        frameNumber -= clvTimecode.seconds * 30;

        clvTimecode.pictureNumber = frameNumber;
    }

    return clvTimecode;
}

// Method to return a description string for the current video format
std::string LdDecodeMetaData::getVideoSystemDescription() const
{
    return getSystemDefaults(videoParameters).name;
}

// Private method to generate a map of the PCM audio data (used by the sourceAudio library)
// Note: That the map unit is "stereo sample pairs"; so each unit represents 2 16-bit samples
// for a total of 4 bytes per unit.
void LdDecodeMetaData::generatePcmAudioMap()
{
    pcmAudioFieldStartSampleMap.clear();
    pcmAudioFieldLengthMap.clear();

    chd::log::debug() << "LdDecodeMetaData::generatePcmAudioMap(): Generating PCM audio map...";

    // Get the number of fields and resize the maps
    int32_t numberOfFields = getVideoParameters().numberOfSequentialFields;
    pcmAudioFieldStartSampleMap.resize(numberOfFields + 1);
    pcmAudioFieldLengthMap.resize(numberOfFields + 1);

    for (int32_t fieldNo = 0; fieldNo < numberOfFields; fieldNo++) {
        // Each audio sample is 16 bit - and there are 2 samples per stereo pair
        pcmAudioFieldLengthMap[fieldNo] = static_cast<int32_t>(getField(fieldNo+1).audioSamples);

        if (fieldNo == 0) {
            // First field starts at 0 units
            pcmAudioFieldStartSampleMap[fieldNo] = 0;
        } else {
            // Every following field's start position is the start+length of the previous
            pcmAudioFieldStartSampleMap[fieldNo] = pcmAudioFieldStartSampleMap[fieldNo - 1] + pcmAudioFieldLengthMap[fieldNo - 1];
        }
    }
}

// Method to get the start sample location of the specified sequential field number
int32_t LdDecodeMetaData::getFieldPcmAudioStart(int32_t sequentialFieldNumber)
{
    if (pcmAudioFieldStartSampleMap.size() < sequentialFieldNumber) return -1;
    // Field numbers are 1 indexed, but our map is 0 indexed
    return pcmAudioFieldStartSampleMap[sequentialFieldNumber - 1];
}

// Method to get the sample length of the specified sequential field number
int32_t LdDecodeMetaData::getFieldPcmAudioLength(int32_t sequentialFieldNumber)
{
    if (pcmAudioFieldLengthMap.size() < static_cast<size_t>(sequentialFieldNumber)) return -1;
    // Field numbers are 1 indexed, but our map is 0 indexed
    return pcmAudioFieldLengthMap[sequentialFieldNumber - 1];
}

}  // namespace chd::metadata
