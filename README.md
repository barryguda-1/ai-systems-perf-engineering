# ai-systems-perf-engineering

Study notes and interactive visualizers for AI systems performance engineering —
GPU software stacks (OS → driver → CUDA → PyTorch), the `torch.compile` pipeline
(TorchDynamo, AOT Autograd, TorchInductor, Triton), and system-level tuning that
keeps GPUs busy.

## Contents

- [`pytorch-cuda-stack-visualizer.html`](pytorch-cuda-stack-visualizer.html) —
  interactive companion to the chapter on the PyTorch/CUDA stack: the layered
  software tower, an animated trace of one Python call becoming many CUDA calls,
  kernel fusion before/after, the training-job pipeline, and GPU-busy timelines.
  Open it directly in any browser (fully self-contained, no dependencies).
