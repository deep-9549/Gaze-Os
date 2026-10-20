# ADR 0001: ONNX Runtime for inference, not PyTorch or TorchScript

**Status:** accepted · **PRD:** 8.3, 10.8

The app runs every network (gaze, gestures, word LM) through ONNX Runtime on
the CPU. PyTorch is used only in `training/`.

- ONNX Runtime is several times faster than eager PyTorch on CPU and supports
  static INT8 quantization, which the latency budget (8 ms for M4) needs.
- The installer stays small: PyTorch alone is larger than the rest of the app.
- TorchScript would still ship libtorch; ONNX also lets the L2CS-Net baseline
  and our own models share one runtime interface (`OnnxGazeEstimator`), selected by
  input and output names plus model metadata (bin count, width, start).

Consequence: every model must pass the export parity check (< 0.05°) before
it can be used, and personalization works on the exported `features` output
instead of fine-tuning the network at runtime (see ADR 0002).
