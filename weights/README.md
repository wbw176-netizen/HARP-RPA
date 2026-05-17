# Weights directory

This directory is intentionally kept out of git except for this placeholder. Model checkpoints and pretrained weights are hosted on Hugging Face:

```bash
hf auth login   # required if the Hugging Face repository is private
hf download bowen12355/HARP-RPA \
  --local-dir weights \
  --exclude README.md \
  --exclude .gitattributes
```

Expected local files after downloading:

```text
weights/
|-- rinalmo_giga_pretrained.pt
|-- sha256sums.txt
|-- HARP-RPA_PRA310_S_5fold/
|   |-- HARP-RPA_fold0.ckpt
|   |-- HARP-RPA_fold1.ckpt
|   |-- HARP-RPA_fold2.ckpt
|   |-- HARP-RPA_fold3.ckpt
|   `-- HARP-RPA_fold4.ckpt
`-- HARP-RPA_PRA310_benchmark_5fold/
    |-- HARP-RPA_fold0.ckpt
    |-- HARP-RPA_fold1.ckpt
    |-- HARP-RPA_fold2.ckpt
    |-- HARP-RPA_fold3.ckpt
    `-- HARP-RPA_fold4.ckpt
```

Verify downloaded files with:

```bash
cd weights
sha256sum -c sha256sums.txt
```

ESMC weights are not included in the HARP-RPA weight release. The code loads ESMC from the EvolutionaryScale pretrained registry by default. For offline use, set a local ESMC checkpoint path before running:

```bash
export ESMC_600M_WEIGHTS=/path/to/esmc_600m_2024_12_v0.pth
```
