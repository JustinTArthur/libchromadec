#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert the project's ONNX NN models to native CoreML ``.mlpackage`` bundles.

These bundles are what ``chd_nn_model_load_from_file`` loads (with
``opts.backend = CHD_NN_COREML``) to reach GPU/ANE on macOS — in particular for
nnTransform3D, whose 3D convolution the
ONNX Runtime CoreML execution provider rejects (it silently runs on CPU). The
generated ``.mlpackage`` files are *not* committed (see .gitignore); regenerate
them from the ONNX models on each machine / in CI.

coremltools' unified converter does not ingest ONNX directly (the ONNX
front-end was removed in coremltools 6). The robust route is
ONNX -> torch.nn.Module (via onnx2torch) -> TorchScript trace -> coremltools.

Dependencies (install into the project-local .venv, never system Python):

    .venv/bin/pip install coremltools onnx2torch torch

NOTE: coremltools tracks specific Python versions; if the project .venv is on a
Python coremltools doesn't yet support, create a dedicated converter venv on a
supported Python just for this script (e.g. python3.11 -m venv .venv-coreml).

Converter coverage (onnx2torch): nnTransform3D chroma_net and both ldzeug2
models (color_cnn and luma_sep) convert and are numerically faithful (rel-RMS
~2e-6 vs ONNX Runtime). This script applies two pre-passes onnx2torch needs:
it folds Constant-node weights into initializers (luma_sep exports its conv
weights as Constant outputs, which onnx2torch's Conv converter can't resolve)
and rewrites Conv auto_pad=SAME_UPPER to explicit pads (onnx2torch can't ingest
auto_pad=SAME_*).

Every package is converted at a FIXED input shape and only runs at that shape:
the TorchScript trace bakes the example dims into the model's dynamic-shape ops,
and a flexible (RangeDim) shape either runs far slower (nnTransform3D's dynamic
batch was ~30x slower) or fails outright at any non-default shape (ldzeug2
raises "Error in dynamically resizing for sequence length (error -7)" — its
Range/ScatterND/Expand ops don't re-derive under a resized input). Convert one
package per runtime shape.

Models (paths per the project's reference notes):
  - nnTransform3D chroma_net*.onnx : input [256,2,4,16,16] -> [256,1,4,16,16];
        the batch is fixed at NNT3D_BATCH (the pipeline's chunk size, which pads
        its last partial chunk up to it). Pick the .onnx whose magnitude scale
        matches the decoder's nnInputMagnitudeScale (1.0 for the v1 series,
        128.0 for true v2).
  - ldzeug2 color_cnn : [1,3,H,W] -> [1,3,H,W]; --height/--width = the decoder's
        modelHeight x fieldWidth. color_cnn runs in field mode, so for NTSC use
        263x910.
  - ldzeug2 luma_sep  : [1,1,H,W] -> [1,1,H,W]; same fixed-shape rule. NTSC
        field = 263x910, frame = 526x910.

Usage:
    .venv/bin/python scripts/convert_coreml.py \
        --model-type nntransform3d --onnx chroma_net.onnx --out chroma_net.mlpackage
    .venv/bin/python scripts/convert_coreml.py \
        --model-type ldzeug-colorcnn --onnx color_cnn.onnx --out color_cnn.mlpackage \
        --height 263 --width 910
"""

import argparse
import sys

# Third-party deps are optional at import time so `--help` works without them;
# convert() reports the install hint if they're missing. coremltools' unified
# converter has no ONNX front-end (removed in coremltools 6), hence the
# ONNX -> torch -> coremltools route.
try:
    import coremltools as ct
    import numpy as np
    import onnx
    import torch
    from onnx import helper
    from onnx2torch import convert as onnx_to_torch
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    ct = np = onnx = torch = helper = onnx_to_torch = None
    _IMPORT_ERROR = exc

# nnTransform3D tile-batch size. MUST match kBatchBlocks in
# src/decoders/nntransform3d/nntransform3d_pipeline_coreml.mm: a *fixed* CoreML
# batch lets the GPU parallelise the conv across the batch (~30x faster than a
# dynamic RangeDim batch, measured on Apple Silicon). The pipeline pads its last
# partial chunk up to this size.
NNT3D_BATCH = 256


# Fixed-shape input per model type. Every model here converts at a *fixed* input
# shape: the TorchScript trace bakes the example dims into the model's
# dynamic-shape ops, so a converted package is valid only at its trace shape. A
# flexible (RangeDim) shape ran ~30x slower for nnTransform3D and fails outright
# for ldzeug2 (CoreML "error -7" at any non-default shape). Convert one package
# per runtime shape. `height`/`width` set the ldzeug2 spatial dims (= the
# decoder's modelHeight x fieldWidth, e.g. NTSC field 263x910); nnTransform3D
# ignores them (its shape is the fixed tile).
def input_shape(model_type, height, width):
    if model_type == "nntransform3d":
        # 3D-FFT tile batch: [N, 2 channels, 4 temporal, 16 y, 16 x]. N is FIXED
        # at NNT3D_BATCH (the pipeline's chunk size) so CoreML parallelises the
        # GPU conv across the batch; a dynamic batch ran ~30x slower. The
        # pipeline pads its last partial chunk up to NNT3D_BATCH.
        shape = (NNT3D_BATCH, 2, 4, 16, 16)
        return ("input", shape, shape)
    if model_type == "ldzeug-colorcnn":
        shape = (1, 3, height, width)
        return ("input", shape, shape)
    if model_type == "ldzeug-lumasep":
        shape = (1, 1, height, width)
        return ("input", shape, shape)
    raise ValueError(f"unknown model type: {model_type}")


def _fold_constant_nodes(model):
    """Move ``Constant`` node outputs into graph initializers.

    onnx2torch's Conv converter looks up weights in ``graph.initializers`` only;
    some exports (e.g. ldzeug2 luma_sep) emit the conv weights as ``Constant``
    node outputs instead, so the converter KeyErrors on the weight tensor name.
    A Constant node is just a static tensor, so folding every value-tensor
    Constant into an initializer is behaviour-preserving and lets the standard
    converters resolve those inputs.
    """
    g = model.graph
    init_names = {i.name for i in g.initializer}
    kept, folded = [], 0
    for node in g.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            value = next((a.t for a in node.attribute if a.name == "value"), None)
            if value is not None and node.output[0] not in init_names:
                tensor = onnx.TensorProto()
                tensor.CopyFrom(value)
                tensor.name = node.output[0]
                g.initializer.append(tensor)
                folded += 1
                continue
        kept.append(node)
    del g.node[:]
    g.node.extend(kept)
    if folded:
        print(f"  folded {folded} Constant node(s) into initializers")


def _resolve_same_padding(model):
    """Rewrite Conv auto_pad=SAME_UPPER/SAME_LOWER to explicit symmetric pads.

    onnx2torch raises NotImplementedError on auto_pad=SAME_*. For the stride-1
    odd-kernel convs in these models, SAME padding is just (k-1)/2 on each side,
    which is what an explicit `pads` attribute expresses.
    """
    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        ap = next((a for a in node.attribute if a.name == "auto_pad"), None)
        if ap is None or ap.s not in (b"SAME_UPPER", b"SAME_LOWER"):
            continue
        kernel = next((list(a.ints) for a in node.attribute if a.name == "kernel_shape"), None)
        strides = next((list(a.ints) for a in node.attribute if a.name == "strides"), None)
        if kernel is None:
            continue
        if strides and any(s != 1 for s in strides):
            raise NotImplementedError("SAME padding with stride>1 needs asymmetric handling")
        node.attribute.remove(ap)
        pads = [(k - 1) // 2 for k in kernel] * 2          # [begin..., end...]
        node.attribute.append(helper.make_attribute("pads", pads))
        rewritten += 1
    if rewritten:
        print(f"  rewrote auto_pad=SAME on {rewritten} Conv node(s) to explicit pads")


def convert(model_type, onnx_path, out_path, compute_units, precision, height, width):
    if _IMPORT_ERROR is not None:
        sys.exit(
            f"missing dependency: {_IMPORT_ERROR}\n"
            "install with: .venv/bin/pip install coremltools onnx2torch torch onnx"
        )

    name, input_dims, example_dims = input_shape(model_type, height, width)

    print(f"loading ONNX: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    _fold_constant_nodes(onnx_model)    # weights-as-Constant -> initializers
    _resolve_same_padding(onnx_model)   # onnx2torch can't ingest auto_pad=SAME_*
    torch_model = onnx_to_torch(onnx_model).eval()

    example = torch.zeros(example_dims, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, example)

    units = {
        "all": ct.ComputeUnit.ALL,
        "cpuandgpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpuonly": ct.ComputeUnit.CPU_ONLY,
    }[compute_units]

    # FLOAT32 is the default. fp16 is per-model: only weights whose input
    # contract keeps every tensor inside fp16 range can take it — of the
    # supported models that is nnTransform3D chroma_net v2 (the ÷128 input
    # scale exists precisely so the spectrum fits fp16; a v1-series model's
    # unscaled magnitudes overflow 65504 and the masks come out NaN). Both
    # ldzeug2 models stay fp32 regardless of scale: fp16 wrecks color_cnn's
    # Range/Gather/ScatterND index math (rel-RMS ~0.9). An fp16 program is
    # also what makes the model ANE-eligible (the ANE runs fp16 only), so
    # convert fp16 when targeting CHD_NN_COREML_ALL.
    prec = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32

    # I/O dtypes are pinned to fp32 in both modes. Left unpinned, an fp16
    # conversion flips the declared I/O descriptors to FLOAT16, which breaks
    # the engine's fp32 outputBackings fast path; pinned, the casts live
    # inside the program and the package presents the same interface as the
    # fp32 one.
    print(f"converting -> {out_path} (compute_units={compute_units}, precision={precision})")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=input_dims, dtype=np.float32)],
        outputs=[ct.TensorType(dtype=np.float32)],
        compute_units=units,
        compute_precision=prec,
        minimum_deployment_target=ct.target.macOS13,
        convert_to="mlprogram",
    )
    mlmodel.save(out_path)
    print("done.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", required=True,
                   choices=["nntransform3d", "ldzeug-colorcnn", "ldzeug-lumasep"])
    p.add_argument("--onnx", required=True, help="source ONNX model path")
    p.add_argument("--out", required=True, help="output .mlpackage path")
    p.add_argument("--compute-units", default="all",
                   choices=["all", "cpuandgpu", "cpuonly"],
                   help="CoreML compute units to bake into the package "
                        "(use cpuandgpu for the nnTransform3D Layer-2 GPU-FFT path)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"],
                   help="compute precision. fp16 is valid ONLY for nnTransform3D "
                        "chroma_net v2 (the ÷128-scale weights) and makes it "
                        "ANE-eligible; every other supported model needs fp32 "
                        "(v1-series chroma_net overflows, ldzeug2 index math breaks)")
    p.add_argument("--height", type=int, default=263,
                   help="ldzeug2 input height = decoder modelHeight; the package "
                        "ONLY runs at this exact shape (color_cnn is field mode → "
                        "NTSC 263). Ignored for nntransform3d.")
    p.add_argument("--width", type=int, default=910,
                   help="ldzeug2 input width = decoder fieldWidth (NTSC 4fsc 910). "
                        "Ignored for nntransform3d.")
    args = p.parse_args()
    convert(args.model_type, args.onnx, args.out, args.compute_units, args.precision,
            args.height, args.width)


if __name__ == "__main__":
    main()
