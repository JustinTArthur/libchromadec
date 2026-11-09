// SPDX-License-Identifier: GPL-3.0-or-later

#include "ldzeug_base.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../common/log.h"
#include "../../nn/inference_engine.h"

namespace chd::decoders::ldzeug {

bool LdzeugDecoderBase::configure(
    const chd::metadata::LdDecodeMetaData::VideoParameters &vp)
{
    if (vp.system != chd::metadata::NTSC) {
        chd::log::fail() << "ldzeug decoders are for NTSC video sources only";
        return false;
    }
    videoParameters = vp;
    return true;
}

void LdzeugDecoderBase::setNnModel(std::shared_ptr<chd::nn::InferenceEngine> engine)
{
    session_ = std::move(engine);
}

void LdzeugDecoderBase::requireOutputShape(const chd::nn::RunResult &result,
                                           const char *decoderName,
                                           int64_t channels, int64_t height, int64_t width)
{
    if (result.count() == 0) {
        throw std::runtime_error(std::string(decoderName) + ": inference produced no outputs");
    }
    const std::vector<int64_t> shape = result.shape(0);
    if (shape.size() == 4 && shape[1] == channels && shape[2] == height && shape[3] == width) {
        return;
    }
    std::string got;
    for (const int64_t dim : shape) {
        if (!got.empty()) got += ", ";
        got += std::to_string(dim);
    }
    throw std::runtime_error(std::string(decoderName) + ": model output shape [" + got +
                             "] does not match the requested [N, " + std::to_string(channels) +
                             ", " + std::to_string(height) + ", " + std::to_string(width) + "]");
}

}  // namespace chd::decoders::ldzeug
