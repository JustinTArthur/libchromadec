// SPDX-License-Identifier: GPL-3.0-or-later
//
// SECAM video-system plumbing: name parsing (SECAM/ME-SECAM), system
// defaults (PAL raster, bell-centre reference frequency), and the open-time
// colour-standard re-declaration for captures whose sidecar says PAL
// (vhs-decode ME-SECAM writes a PAL sidecar).

#include <chromadec/video.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/metadata/core.h"

namespace fs = std::filesystem;

namespace {

// 625-line PAL-geometry sidecar in the legacy ld-decode JSON schema, the
// shape vhs-decode writes for an ME-SECAM capture (nothing in it says SECAM).
const char *kPalJsonSidecar = R"({
  "videoParameters": {
    "activeVideoEnd": 1107,
    "activeVideoStart": 185,
    "black16bIre": 16384,
    "colourBurstEnd": 138,
    "colourBurstStart": 98,
    "fieldHeight": 313,
    "fieldWidth": 1135,
    "isMapped": false,
    "isSubcarrierLocked": false,
    "isWidescreen": false,
    "numberOfSequentialFields": 4,
    "sampleRate": 17734475,
    "system": "PAL",
    "white16bIre": 54016
  },
  "fields": [
    {"seqNo": 1, "isFirstField": true,  "syncConf": 100, "medianBurstIRE": 25.0},
    {"seqNo": 2, "isFirstField": false, "syncConf": 100, "medianBurstIRE": 25.0},
    {"seqNo": 3, "isFirstField": true,  "syncConf": 100, "medianBurstIRE": 25.0},
    {"seqNo": 4, "isFirstField": false, "syncConf": 100, "medianBurstIRE": 25.0}
  ]
})";

bool writeFakeTbc(const std::string &path, int32_t fieldWidth, int32_t fieldHeight,
                  int32_t numFields) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const std::vector<uint16_t> buffer(fieldWidth * fieldHeight, 0);
    for (int32_t i = 0; i < numFields; i++) {
        f.write(reinterpret_cast<const char *>(buffer.data()), buffer.size() * 2);
    }
    return f.good();
}

bool writeText(const std::string &path, const char *text) {
    std::ofstream f(path);
    if (!f) return false;
    f << text;
    return f.good();
}

#define REQUIRE(cond)                                                            \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond \
                      << "\n";                                                   \
            return 1;                                                            \
        }                                                                        \
    } while (0)

int testParseVideoSystemName() {
    using chd::metadata::VideoSystem;
    using chd::metadata::parseVideoSystemName;

    VideoSystem system = chd::metadata::NTSC;
    REQUIRE(parseVideoSystemName("SECAM", system));
    REQUIRE(system == chd::metadata::SECAM);
    REQUIRE(parseVideoSystemName("mesecam", system));
    REQUIRE(system == chd::metadata::SECAM);
    REQUIRE(parseVideoSystemName("ME-SECAM", system));
    REQUIRE(system == chd::metadata::SECAM);

    // Neighbouring aliases keep their existing mappings.
    REQUIRE(parseVideoSystemName("PAL", system));
    REQUIRE(system == chd::metadata::PAL);
    REQUIRE(parseVideoSystemName("PAL N", system));
    REQUIRE(system == chd::metadata::PAL);
    REQUIRE(parseVideoSystemName("MPAL", system));
    REQUIRE(system == chd::metadata::PAL_M);
    return 0;
}

int testOverrideVideoSystem() {
    chd::metadata::LdDecodeMetaData meta;
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::PAL;
    vp.fSC = (283.75 * 15625) + 25;
    vp.fieldWidth = 1135;
    vp.fieldHeight = 313;
    vp.sampleRate = 17734475.0;
    vp.black16bIre = 16384;
    vp.white16bIre = 54016;
    vp.blanking16bIre = 16384;
    vp.firstActiveFrameLine = 44;
    vp.lastActiveFrameLine = 619;
    meta.setVideoParameters(vp);

    meta.overrideVideoSystem(chd::metadata::SECAM);
    const auto &out = meta.getVideoParameters();
    REQUIRE(out.system == chd::metadata::SECAM);
    REQUIRE(out.fSC == 4286000.0);
    // SECAM shares the PAL raster, so the active-line ranges hold.
    REQUIRE(out.firstActiveFrameLine == 44);
    REQUIRE(out.lastActiveFrameLine == 619);
    REQUIRE(out.firstActiveFieldLine() == 22);
    REQUIRE(out.lastActiveFieldLine() == 309);
    return 0;
}

int testOpenTimeRedeclaration(const fs::path &dir) {
    const std::string tbc     = (dir / "mesecam.tbc").string();
    const std::string sidecar = (dir / "mesecam.tbc.json").string();
    REQUIRE(writeFakeTbc(tbc, 1135, 313, 4));
    REQUIRE(writeText(sidecar, kPalJsonSidecar));

    // Without an override the sidecar's PAL declaration stands.
    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), nullptr, nullptr, &video) == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(video, &info) == CHD_OK);
    REQUIRE(info.standard == CHD_STD_PAL);
    chd_video_free(video);

    // Re-declared SECAM: standard flips, the reference frequency follows the
    // SECAM defaults, and the 625-line active region is unchanged.
    chd_video_params_t params{};
    params.standard = CHD_STD_SECAM;
    video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), nullptr, &params, &video) == CHD_OK);
    REQUIRE(chd_video_get_info(video, &info) == CHD_OK);
    REQUIRE(info.standard == CHD_STD_SECAM);
    REQUIRE(info.fsc_hz == 4286000.0);
    REQUIRE(info.first_active_frame_line == 44);
    REQUIRE(info.last_active_frame_line == 619);
    REQUIRE(info.num_frames == 2);
    chd_video_free(video);

    // A cross-line-standard re-declaration is rejected.
    params.standard = CHD_STD_NTSC;
    video = nullptr;
    REQUIRE(chd_video_open_composite(tbc.c_str(), nullptr, &params, &video) != CHD_OK);
    REQUIRE(video == nullptr);

    fs::remove(tbc);
    fs::remove(sidecar);
    return 0;
}

int testOpenYcRedeclaration(const fs::path &dir) {
    // Y/C pair with one shared sidecar, the layout vhs-decode writes.
    const std::string luma    = (dir / "pair.tbc").string();
    const std::string chroma  = (dir / "pair_chroma.tbc").string();
    const std::string sidecar = (dir / "pair.tbc.json").string();
    REQUIRE(writeFakeTbc(luma, 1135, 313, 4));
    REQUIRE(writeFakeTbc(chroma, 1135, 313, 4));
    REQUIRE(writeText(sidecar, kPalJsonSidecar));

    chd_video_params_t params{};
    params.standard = CHD_STD_SECAM;
    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_yc(luma.c_str(), chroma.c_str(), nullptr, &params, &video)
            == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(video, &info) == CHD_OK);
    REQUIRE(info.standard == CHD_STD_SECAM);
    chd_video_free(video);

    fs::remove(luma);
    fs::remove(chroma);
    fs::remove(sidecar);
    return 0;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "chd_secam_metadata_test";
    fs::create_directories(dir);

    int rc = testParseVideoSystemName();
    if (rc == 0) rc = testOverrideVideoSystem();
    if (rc == 0) rc = testOpenTimeRedeclaration(dir);
    if (rc == 0) rc = testOpenYcRedeclaration(dir);

    fs::remove_all(dir);
    if (rc == 0) std::cout << "PASS\n";
    return rc;
}
