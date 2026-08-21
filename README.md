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

- [`numa-pinned-memory-visualizer.html`](numa-pinned-memory-visualizer.html) —
  interactive companion to the NUMA-friendly allocation & memory-pinning section:
  a two-socket server diagram, a placement lab that animates where each batch
  travels (and counts UPI hops), an annotated `numactl` command with fork-vs-spawn
  inheritance, a pinned-vs-pageable H2D transfer race, GPUDirect paths, and a
  `pin_memory` / `ulimit -l` (memlock) lab, plus a field guide (`lscpu`, `numactl -H`,
  `nvidia-smi topo -m`, `numastat`), real-world bandwidth figures, and gotchas
  (strict vs preferred binding, `--cpuset-mems`, the Python 3.14 forkserver change).
  Self-contained, no dependencies.
