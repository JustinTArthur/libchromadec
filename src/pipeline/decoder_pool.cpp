// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    decoderpool.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2021 Phillip Blucas
    Copyright (C) 2021 Adam Sampson

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#include "decoder_pool.h"

#include "../common/error_state.h"

#include <algorithm>
#include <thread>

#include "../common/log.h"
#include "../output/component_frame.h"

namespace chd::pipeline {

namespace {
template <typename T>
T qMin(T a, T b) { return std::min(a, b); }
template <typename T>
T qMax(T a, T b) { return std::max(a, b); }

// Worker loop driving one Decoder synchronously: pulls a batch of input
// fields from the pool, calls Decoder::decodeFrames on them, converts the
// resulting ComponentFrames to OutputFrames via the shared OutputWriter,
// and pushes them back to the pool. Replaces the legacy DecoderThread::run.
void workerLoop(DecoderPool *pool) {
    auto &decoder = pool->getDecoder();
    auto &outputWriter = pool->getOutputWriter();
    auto &abort = pool->getAbort();

    while (!abort.load()) {
        int32_t startFrameNumber = 0;
        int32_t startIndex = 0;
        int32_t endIndex = 0;
        std::vector<chd::decoders::SourceField> inputFields;

        if (!pool->getInputFrames(startFrameNumber, inputFields, startIndex, endIndex)) {
            // End of input
            return;
        }

        const int32_t numFrames = (endIndex - startIndex) / 2;
        std::vector<chd::output::ComponentFrame> componentFrames(numFrames);

        try {
            decoder.decodeFrames(inputFields, startIndex, endIndex, componentFrames);
        } catch (const std::exception &e) {
            chd::log::fail() << "Decoder worker failed:" << e.what();
            pool->recordAbortReason(std::string("decoder worker failed: ") + e.what());
            abort.store(true);
            return;
        }

        std::vector<chd::output::OutputFrame> outputFrames(numFrames);
        for (int32_t i = 0; i < numFrames; i++) {
            outputWriter.convert(componentFrames[i], outputFrames[i]);
        }

        if (!pool->putOutputFrames(startFrameNumber, outputFrames)) {
            abort.store(true);
            return;
        }
    }
}
}  // namespace

DecoderPool::DecoderPool(chd::decoders::Decoder &_decoder, std::string _inputFileName,
                         chd::metadata::LdDecodeMetaData &_ldDecodeMetaData,
                         chd::output::OutputWriter::Configuration &_outputConfig, std::string _outputFileName,
                         int32_t _startFrame, int32_t _length, int32_t _maxThreads)
    : decoder(_decoder), inputFileName(_inputFileName),
      outputConfig(_outputConfig), outputFileName(_outputFileName),
      startFrame(_startFrame), length(_length), maxThreads(_maxThreads),
      abort(false), ldDecodeMetaData(_ldDecodeMetaData)
{
}

chd::decoders::Decoder& DecoderPool::getDecoder() { return decoder; }

void DecoderPool::recordAbortReason(std::string reason)
{
    std::lock_guard<std::mutex> lock(abortReasonMutex);
    if (abortReason.empty()) abortReason = std::move(reason);
}

bool DecoderPool::process()
{
    chd::metadata::LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    // Configure the OutputWriter, adjusting videoParameters
    outputWriter.updateConfiguration(videoParameters, outputConfig);
    outputWriter.printOutputInfo();

    // Configure the decoder, and check that it can accept this video
    if (!decoder.configure(videoParameters)) {
        return false;
    }

    // Get the decoder's lookbehind/lookahead requirements
    decoderLookBehind = decoder.getLookBehind();
    decoderLookAhead = decoder.getLookAhead();

    // Open the source video file
    if (!sourceVideo.open(inputFileName, videoParameters.fieldWidth * videoParameters.fieldHeight)) {
        // Could not open source video file
        chd::log::fail() << "Unable to open ld-decode video file";
        return false;
    }

    // If no startFrame parameter was specified, set the start frame to 1
    if (startFrame == -1) startFrame = 1;

    if (startFrame > ldDecodeMetaData.getNumberOfFrames()) {
        chd::log::fail() << "Specified start frame is out of bounds, only" << ldDecodeMetaData.getNumberOfFrames() << "frames available";
        return false;
    }

    // If no length parameter was specified set the length to the number of available frames
    if (length == -1) {
        length = ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
    } else {
        if (length + (startFrame - 1) > ldDecodeMetaData.getNumberOfFrames()) {
            chd::log::info() << "Specified length of" << length << "exceeds the number of available frames, setting to" << ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
            length = ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
        }
    }

    // Open the output file. Stdout is not supported here (libchromadec
    // is a library; the consumer can pipe a regular file path).
    if (outputFileName == "-") {
        chd::log::fail() << "Stdout output is not supported by libchromadec";
        sourceVideo.close();
        return false;
    }
    targetVideo.open(outputFileName, std::ios::binary);
    if (!targetVideo.is_open()) {
        chd::log::fail() << "Could not open" << outputFileName << "for output";
        sourceVideo.close();
        return false;
    }

    // Write the stream header (if there is one)
    const std::string streamHeader = outputWriter.getStreamHeader();
    if (!streamHeader.empty()) {
        targetVideo.write(streamHeader.data(), streamHeader.size());
        if (!targetVideo.good()) {
            chd::log::fail() << "Writing to the output video file failed";
            return false;
        }
    }

    chd::log::info() << "Using" << maxThreads << "threads";
    chd::log::info() << "Processing from start frame #" << startFrame << "with a length of" << length << "frames";

    // Initialise processing state
    inputFrameNumber = startFrame;
    outputFrameNumber = startFrame;
    lastFrameNumber = length + (startFrame - 1);
    totalTimerStart = std::chrono::steady_clock::now();

    // Start a vector of filtering threads to process the video
    std::vector<std::thread> threads;
    threads.reserve(maxThreads);
    for (int32_t i = 0; i < maxThreads; i++) {
        threads.emplace_back(workerLoop, this);
    }

    // Wait for the workers to finish
    for (auto &t : threads) {
        t.join();
    }

    // Did any of the threads abort?
    if (abort.load()) {
        std::lock_guard<std::mutex> lock(abortReasonMutex);
        if (!abortReason.empty()) chd::detail::set_last_error(abortReason);
        sourceVideo.close();
        targetVideo.close();
        return false;
    }

    // Check we've processed all the frames, now the workers have finished
    if (inputFrameNumber != (lastFrameNumber + 1) || outputFrameNumber != (lastFrameNumber + 1)
        || !pendingOutputFrames.empty()) {
        chd::log::fail() << "Incorrect state at end of processing";
        sourceVideo.close();
        targetVideo.close();
        return false;
    }

    const auto elapsed = std::chrono::steady_clock::now() - totalTimerStart;
    const double totalSecs = std::chrono::duration<double>(elapsed).count();
    chd::log::info() << "Processing complete -" << length << "frames in" << totalSecs << "seconds (" <<
               length / totalSecs << "FPS )";

    // Close the source video
    sourceVideo.close();

    // Close the target video
    targetVideo.close();

    return true;
}

bool DecoderPool::getInputFrames(int32_t &startFrameNumber, std::vector<chd::decoders::SourceField> &fields, int32_t &startIndex, int32_t &endIndex)
{
    std::lock_guard<std::mutex> locker(inputMutex);

    // Work out a reasonable batch size to provide work for all threads.
    // This assumes that the synchronisation to get a new batch is less
    // expensive than computing a single frame, so a batch size of 1 is
    // reasonable.
    const int32_t maxBatchSize = qMin(DEFAULT_BATCH_SIZE, qMax(1, length / maxThreads));

    // Work out how many frames will be in this batch
    int32_t batchFrames = qMin(maxBatchSize, lastFrameNumber + 1 - inputFrameNumber);
    if (batchFrames == 0) {
        // No more input frames
        return false;
    }

    // Advance the frame number
    startFrameNumber = inputFrameNumber;
    inputFrameNumber += batchFrames;

    // Load the fields
    chd::decoders::SourceField::loadFields(sourceVideo, ldDecodeMetaData,
                            startFrameNumber, batchFrames, decoderLookBehind, decoderLookAhead,
                            fields, startIndex, endIndex);

    return true;
}

bool DecoderPool::putOutputFrames(int32_t startFrameNumber, const std::vector<chd::output::OutputFrame> &outputFrames)
{
    std::lock_guard<std::mutex> locker(outputMutex);

    for (size_t i = 0; i < outputFrames.size(); i++) {
        if (!putOutputFrame(startFrameNumber + static_cast<int32_t>(i), outputFrames[i])) {
            return false;
        }
    }

    return true;
}

// Write one output frame. You must hold outputMutex to call this.
//
// The worker threads will complete frames in an arbitrary order, so we can't
// just write the frames to the output file directly. Instead, we keep a map of
// frames that haven't yet been written; when a new frame comes in, we check
// whether we can now write some of them out.
//
// Returns true on success, false on failure.
bool DecoderPool::putOutputFrame(int32_t frameNumber, const chd::output::OutputFrame &outputFrame)
{
    // Put this frame into the map
    pendingOutputFrames[frameNumber] = outputFrame;

    // Write out as many frames as possible
    while (pendingOutputFrames.count(outputFrameNumber) > 0) {
        const chd::output::OutputFrame& outputData = pendingOutputFrames[outputFrameNumber];

        // Write the frame header (if there is one)
        const std::string frameHeader = outputWriter.getFrameHeader();
        if (!frameHeader.empty()) {
            targetVideo.write(frameHeader.data(), frameHeader.size());
            if (!targetVideo.good()) {
                chd::log::fail() << "Writing to the output video file failed";
                return false;
            }
        }

        // Write the frame data
        targetVideo.write(reinterpret_cast<const char *>(outputData.data()), outputData.size() * 2);
        if (!targetVideo.good()) {
            chd::log::fail() << "Writing to the output video file failed";
            return false;
        }

        pendingOutputFrames.erase(outputFrameNumber);
        outputFrameNumber++;

        const int32_t outputCount = outputFrameNumber - startFrame;
        if ((outputCount % 32) == 0) {
            // Show an update to the user
            const auto elapsed = std::chrono::steady_clock::now() - totalTimerStart;
            const double secs = std::chrono::duration<double>(elapsed).count();
            const double fps = outputCount / secs;
            chd::log::info() << outputCount << "frames processed -" << fps << "FPS";
        }
    }

    return true;
}

}  // namespace chd::pipeline
