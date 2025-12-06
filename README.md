# Unsloth Mojo Experiments

A scratchpad for **Mojo/MAX kernel experiments**, focusing on LLM training/inference primitives with comparisons against PyTorch, Triton, and CUDA implementations.

The main experiment is a **bitsandbytes-style NF4 → fp16 dequantization kernel**, inspired by [Unsloth's hiring puzzle](https://github.com/unslothai/unsloth/wiki) — implemented entirely in **Mojo**.

> **TL;DR** On an NVIDIA L4, the best Mojo kernel runs the full Unsloth `test_dequantize` workload in **~2.47s** vs **~3.02s** for Unsloth's CUDA `fast_dequantize` and **~5.32s** for the Triton reference.

---

## Contents

- **`Unsloth_Mojo_Puzzles.ipynb`** — Colab notebook with:
  - NF4 → fp16 dequant kernels (2D tiled + packed stores, warp-per-block variants)
  - CPU reference implementation of bitsandbytes NF4 + double quant
  - Benchmark harness mirroring Unsloth's `test_dequantize` (9000 dequant calls across 3 configs)
  - Correctness tests and timing summaries

---

## How to Run

### Colab

1. Open the notebook in Colab with a GPU runtime (T4 or L4)
2. Run cells in order — setup, kernels, reference, tests, benchmarks
3. View per-config times and speedup vs Triton reference

### Local

Run the notebook with Jupyter, or extract the Mojo kernel code into a `.mojo` file and run with:

```bash
modular run your_kernel.mojo
```

---

## Repo Structure

```
├── Unsloth_Mojo_Puzzles.ipynb  # Primary playground
├── README.md
└── LICENSE
```

Future additions may include `kernels/` and `benchmarks/` directories.

---

## Roadmap

- **QLoRA with FSDP2** — Mojo/Max-accelerated training components
- **Memory efficient backprop** — Custom kernels to avoid `[B,T,V]` logits materialization
- **Additional kernels** — Attention, matmuls, nvFP4/nvFP8 experiments

---

## Acknowledgements

- **[Unsloth](https://unsloth.ai/)** — For the puzzles and benchmark harness
- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** — For the NF4 quantization scheme
- **[Modular](https://www.modular.com/)** — For Mojo and the MAX GPU backend

*Not affiliated with Unsloth or Modular — just experimenting with Mojo kernels.*
