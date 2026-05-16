# Weights directory

This directory is intentionally kept out of git except for this placeholder.

Expected local files:

```text
weights/rinalmo_giga_pretrained.pt
weights/HARP-RPA_PRA310_S_5fold/HARP-RPA_fold0.ckpt
weights/HARP-RPA_PRA310_S_5fold/HARP-RPA_fold1.ckpt
weights/HARP-RPA_PRA310_S_5fold/HARP-RPA_fold2.ckpt
weights/HARP-RPA_PRA310_S_5fold/HARP-RPA_fold3.ckpt
weights/HARP-RPA_PRA310_S_5fold/HARP-RPA_fold4.ckpt
weights/HARP-RPA_PRA310_benchmark_5fold/HARP-RPA_fold0.ckpt
weights/HARP-RPA_PRA310_benchmark_5fold/HARP-RPA_fold1.ckpt
weights/HARP-RPA_PRA310_benchmark_5fold/HARP-RPA_fold2.ckpt
weights/HARP-RPA_PRA310_benchmark_5fold/HARP-RPA_fold3.ckpt
weights/HARP-RPA_PRA310_benchmark_5fold/HARP-RPA_fold4.ckpt
```

Download released weights from Hugging Face:

```bash
hf download bowen12355/HARP-RPA --local-dir weights
```

For ESMC, either let the EvolutionaryScale `esm` package load the pretrained
model from its registry/cache, or set a local checkpoint path before running:

```bash
export ESMC_600M_WEIGHTS=/path/to/esmc_600m_2024_12_v0.pth
```
