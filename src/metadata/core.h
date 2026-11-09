/******************************************************************************
 * lddecodemetadata.h
 * ld-decode-tools TBC library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2025 Simon Inns
 * SPDX-FileCopyrightText: 2022 Ryan Holtz
 * SPDX-FileCopyrightText: 2022-2023 Adam Sampson
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#ifndef CHD_METADATA_CORE_H
#define CHD_METADATA_CORE_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "dropouts.h"

namespace chd::metadata {

class JsonReader;
class JsonWriter;
class SqliteReader;
class SqliteWriter;

// The video system (combination of a line standard and a colour standard)
// Note: If you update this, be sure to update VIDEO_SYSTEM_DEFAULTS also
enum VideoSystem {
    PAL = 0,    // 625-line PAL
    NTSC,       // 525-line NTSC
    PAL_M,      // 525-line PAL
    SECAM,      // 625-line SECAM
};

bool parseVideoSystemName(std::string name, VideoSystem &system);

class LdDecodeMetaData
{

public:
    // VBI Metadata definition
    struct Vbi {
        bool inUse = false;
        std::array<int32_t, 3> vbiData { 0, 0, 0 };

        void read(SqliteReader &reader, int captureId, int fieldId);
        void write(SqliteWriter &writer, int captureId, int fieldId) const;
        void read(JsonReader &reader);
    };

    // Video metadata definition
    struct VideoParameters {
        // -- Members stored in the metadata --

        int32_t numberOfSequentialFields = -1;

        VideoSystem system = NTSC;
        bool isSubcarrierLocked = false;
        bool isWidescreen = false;

        int32_t colourBurstStart = -1;
        int32_t colourBurstEnd = -1;
        int32_t activeVideoStart = -1;
        int32_t activeVideoEnd = -1;

        int32_t white16bIre = -1;
        int32_t black16bIre = -1;
        int32_t blanking16bIre = -1;

        int32_t fieldWidth = -1;
        int32_t fieldHeight = -1;
        double sampleRate = -1.0;

        bool isMapped = false;
        std::string tapeFormat = "";

        std::string gitBranch;
        std::string gitCommit;

        // -- Members set by the library --

        // Colour subcarrier frequency in Hz
        double fSC = -1.0;

        // The range of active lines within an interlaced frame, 0-indexed and
        // inclusive on both ends (the last value is the last active line, not
        // one past it).
        int32_t firstActiveFrameLine = -1;
        int32_t lastActiveFrameLine = -1;

        // The equivalent active line range within a single field, derived from
        // the frame-line crop above so there is a single source of truth. The
        // dropout corrector is the only consumer.
        int32_t firstActiveFieldLine() const { return (firstActiveFrameLine + 1) / 2; }
        int32_t lastActiveFieldLine()  const { return lastActiveFrameLine / 2; }

        // The source's picture window (half-open), unaffected by caller crops.
        // -1 until resolved: a source that knows it sets it, otherwise it is
        // the active window the source reports.
        int32_t pictureStart = -1;
        int32_t pictureEnd = -1;

        // The horizontal window tile-based NN decoders read (half-open): the
        // picture window, widened by any crop the caller set. -1 until resolved.
        int32_t tileReadStart = -1;
        int32_t tileReadEnd = -1;

        // Flags if our data has been initialized yet
        bool isValid = false;

        void read(SqliteReader &reader, int captureId);
        void write(SqliteWriter &writer, int captureId) const;
        void read(JsonReader &reader);
    };

    // Specification for customising the range of active lines in VideoParameters.
    // -1 for any of these means to use the default for the standard.
    struct LineParameters {
        int32_t firstActiveFrameLine = -1;
        int32_t lastActiveFrameLine = -1;

        void applyTo(VideoParameters &videoParameters);
    };

    // VITS metrics metadata definition
    struct VitsMetrics {
        bool inUse = false;
        double wSNR = 0.0;
        double bPSNR = 0.0;

        void read(SqliteReader &reader, int captureId, int fieldId);
        void write(SqliteWriter &writer, int captureId, int fieldId) const;
        void read(JsonReader &reader);
    };

    // NTSC Specific metadata definition
    struct ClosedCaption;
    struct Ntsc {
        bool inUse = false;
        bool isFmCodeDataValid = false;
        int32_t fmCodeData = 0;
        bool fieldFlag = false;
        bool isVideoIdDataValid = false;
        int32_t videoIdData = 0;
        bool whiteFlag = false;

        void read(SqliteReader &reader, int captureId, int fieldId, ClosedCaption &closedCaption);
        void write(SqliteWriter &writer, int captureId, int fieldId) const;
        void read(JsonReader &reader, ClosedCaption &closedCaption);
    };

    // VITC timecode definition
    struct Vitc {
        bool inUse = false;

        // Just the VITC data, without the sync bits or CRC.
        // vitcData[0]'s LSB is bit 2; vitcData[7]'s MSB is bit 79.
        std::array<int32_t, 8> vitcData;

        void read(SqliteReader &reader, int captureId, int fieldId);
        void write(SqliteWriter &writer, int captureId, int fieldId) const;
        void read(JsonReader &reader);
    };

    // Closed Caption definition
    struct ClosedCaption {
        bool inUse = false;

        int32_t data0 = -1;
        int32_t data1 = -1;

        void read(SqliteReader &reader, int captureId, int fieldId);
        void write(SqliteWriter &writer, int captureId, int fieldId) const;
        void read(JsonReader &reader);
    };

    // PCM sound metadata definition
    struct PcmAudioParameters {
        double sampleRate = -1.0;
        bool isLittleEndian = false;
        bool isSigned = false;
        int32_t bits = -1;

        // Flags if our data has been initialized yet
        bool isValid = false;

        void read(SqliteReader &reader, int captureId);
        void write(SqliteWriter &writer, int captureId) const;
        void read(JsonReader &reader);
    };

    // Field metadata definition
    struct Field {
        int32_t seqNo = 0;   // Note: This is the unique primary-key
        bool isFirstField = false;
        int32_t syncConf = 0;
        double medianBurstIRE = 0.0;
        int32_t fieldPhaseID = -1;
        int32_t audioSamples = -1;

        VitsMetrics vitsMetrics;
        Vbi vbi;
        Ntsc ntsc;
        Vitc vitc;
        ClosedCaption closedCaption;
        DropOuts dropOuts;
        bool pad = false;

        double diskLoc = -1;
        int64_t fileLoc = -1;
        int32_t decodeFaults = -1;
        int32_t efmTValues = -1;

        void read(SqliteReader &reader, int captureId);
        void write(SqliteWriter &writer, int captureId) const;
        void read(JsonReader &reader);
    };

    // CLV timecode (used by frame number conversion methods)
    struct ClvTimecode {
        int32_t hours;
        int32_t minutes;
        int32_t seconds;
        int32_t pictureNumber;
    };

    LdDecodeMetaData();

    // Prevent copying or assignment
    LdDecodeMetaData(const LdDecodeMetaData &) = delete;
    LdDecodeMetaData& operator=(const LdDecodeMetaData &) = delete;

    void clear();
    bool read(std::string fileName);
    bool write(std::string fileName) const;
    void readFields(SqliteReader &reader, int captureId);
    void writeFields(SqliteWriter &writer, int captureId) const;

    const VideoParameters &getVideoParameters();
    void setVideoParameters(const VideoParameters &videoParameters);

    const PcmAudioParameters &getPcmAudioParameters();
    void setPcmAudioParameters(const PcmAudioParameters &pcmAudioParam);

    // Handle line parameters
    void processLineParameters(LdDecodeMetaData::LineParameters &_lineParameters);

    // Re-declare the video system after a successful read/set (e.g. a SECAM
    // capture whose sidecar says PAL): rederives the system defaults and
    // revalidates the active-line ranges against the new system's bounds.
    void overrideVideoSystem(VideoSystem system);

    // Get field metadata
    const Field &getField(int32_t sequentialFieldNumber);
    const VitsMetrics &getFieldVitsMetrics(int32_t sequentialFieldNumber);
    const Vbi &getFieldVbi(int32_t sequentialFieldNumber);
    const Ntsc &getFieldNtsc(int32_t sequentialFieldNumber);
    const Vitc &getFieldVitc(int32_t sequentialFieldNumber);
    const ClosedCaption &getFieldClosedCaption(int32_t sequentialFieldNumber);
    const DropOuts &getFieldDropOuts(int32_t sequentialFieldNumber);

    // Set field metadata
    void updateField(const Field &field, int32_t sequentialFieldNumber);
    void updateFieldVitsMetrics(const LdDecodeMetaData::VitsMetrics &vitsMetrics, int32_t sequentialFieldNumber);
    void updateFieldVbi(const LdDecodeMetaData::Vbi &vbi, int32_t sequentialFieldNumber);
    void updateFieldNtsc(const LdDecodeMetaData::Ntsc &ntsc, int32_t sequentialFieldNumber);
    void updateFieldVitc(const LdDecodeMetaData::Vitc &vitc, int32_t sequentialFieldNumber);
    void updateFieldClosedCaption(const LdDecodeMetaData::ClosedCaption &closedCaption, int32_t sequentialFieldNumber);
    void updateFieldDropOuts(const DropOuts &dropOuts, int32_t sequentialFieldNumber);
    void clearFieldDropOuts(int32_t sequentialFieldNumber);

    void appendField(const Field &field);

    void setNumberOfFields(int32_t numberOfFields);
    int32_t getNumberOfFields();
    // Frame pairing. `firstFieldFirst` is the caller's pairing order, not
    // read from this object: a decoder resolves it per source (the source's
    // as-opened order, swapped by reverse_field_order), so decoders sharing
    // one opened video can pair its fields differently.
    int32_t getNumberOfFrames(bool firstFieldFirst);
    int32_t getFirstFieldNumber(int32_t frameNumber, bool firstFieldFirst);
    int32_t getSecondFieldNumber(int32_t frameNumber, bool firstFieldFirst);

    // The pairing order the source was opened with: true for an ld-decode
    // sidecar, the caller's is_second_field_first override for synthesized
    // CVBS metadata. Set only while opening.
    void setIsFirstFieldFirst(bool flag);
    bool getIsFirstFieldFirst();

    int32_t convertClvTimecodeToFrameNumber(LdDecodeMetaData::ClvTimecode clvTimeCode);
    LdDecodeMetaData::ClvTimecode convertFrameNumberToClvTimecode(int32_t clvFrameNumber);

    // PCM Analogue audio helper methods
    int32_t getFieldPcmAudioStart(int32_t sequentialFieldNumber);
    int32_t getFieldPcmAudioLength(int32_t sequentialFieldNumber);

    // Video system helper methods
    std::string getVideoSystemDescription() const;

private:
    bool isFirstFieldFirst;
    VideoParameters videoParameters;
    PcmAudioParameters pcmAudioParameters;
    std::vector<Field> fields;
    std::vector<int32_t> pcmAudioFieldStartSampleMap;
    std::vector<int32_t> pcmAudioFieldLengthMap;

    bool readSqliteImpl(const std::string &fileName);
    bool readJsonImpl(const std::string &fileName);

    void initialiseVideoSystemParameters();
    int32_t getFieldNumber(int32_t frameNumber, int32_t field, bool firstFieldFirst);
    void generatePcmAudioMap();
};

} // namespace chd::metadata

#endif // CHD_METADATA_CORE_H
