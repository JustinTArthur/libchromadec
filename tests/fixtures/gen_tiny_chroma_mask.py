#!/usr/bin/env python3
"""Generate a tiny ONNX model with chroma_net's input and output shapes.

This is NOT a real chroma-decoding model. It maps [N,2,4,16,16] magnitudes to a
[N,1,4,16,16] mask, so the nnTransform3D tests can run real tile passes without
the real weights. The mask depends on the input, which is all those tests need.

Regenerate (committed output is ``tiny_chroma_mask.onnx`` alongside this script)
with the same venv steps as ``gen_tiny_onnx.py``.
"""
import os

import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2, 4, 16, 16])
y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1, 4, 16, 16])
nodes = [
    helper.make_node("ReduceMean", inputs=["input"], outputs=["mean"], axes=[1], keepdims=1),
    helper.make_node("Sigmoid", inputs=["mean"], outputs=["output"]),
]
graph = helper.make_graph(nodes, "tiny_chroma_mask", [x], [y])

model = helper.make_model(
    graph,
    producer_name="chromadec-tests",
    opset_imports=[helper.make_opsetid("", 13)],
)
# Pin a conservative IR version so older ONNX Runtime builds in CI accept it.
model.ir_version = 8
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_chroma_mask.onnx")
onnx.save(model, out_path)
print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
