public_headers = files(
  'chromadec/calibration.h',
  'chromadec/chromadec.h',
  'chromadec/decoder.h',
  'chromadec/dropout.h',
  'chromadec/enum.h',
  'chromadec/errors.h',
  'chromadec/frame.h',
  'chromadec/log.h',
  'chromadec/nn.h',
  'chromadec/pipeline.h',
  'chromadec/types.h',
  'chromadec/video.h',
)

install_headers(public_headers, subdir: 'chromadec')

# Generate version.h from version.h.in. Done at this subdir level so output
# lands at build/include/chromadec/version.h and <chromadec/version.h> works.
subdir('chromadec')

# Public include path covers both the source headers (include/) and the
# generated build/include/.
public_inc = include_directories('.')
# version_inc is set by include/chromadec/meson.build (= build/include).
