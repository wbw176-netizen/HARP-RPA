# HARP-RPA

HARP-RPA is a protein-RNA binding affinity prediction framework built around heterogeneous residue-pair modeling. It combines pretrained protein and RNA sequence encoders with explicit protein/RNA/interface pair construction, a relation-aware HeteroPairFormer backbone, and an interface-aware readout for complex-level binding free-energy prediction.

The current main model configuration is:

```yaml
./config/models/harp_rpa.yml
```

It uses:

- ESMC-600M as the protein encoder.
- RiNALMo as the RNA encoder.
- Heterogeneous token assembly with complex, protein, and RNA special tokens.
- Heterogeneous residue-pair construction for P-P, R-R, P-R, and special-token pair relations.
- HeteroPairFormer for joint single-token and pair-representation updates.
- `token_interface` pooling, combining the complex token with an interface-aware P-R pair readout.
- Label standardization during affinity training.
- Dual-depth fine-tuning: shallow LoRA and upper-layer fine-tuning for language-model adaptation.

## Installation

Create the recommended conda environment:

```bash
conda env create -f environment.yml
conda activate harp_rpa
```

## Model Weights

HARP-RPA uses ESMC as the default protein encoder. ESMC weights are not included in this repository. By default, the code loads ESMC from the EvolutionaryScale pretrained registry. For offline use, set a local ESMC checkpoint path before running:

```bash
export ESMC_600M_WEIGHTS=/path/to/esmc_600m_2024_12_v0.pth
```

The released HARP-RPA checkpoints and RiNALMo pretrained weights are hosted on Hugging Face:

```bash
hf auth login   # required if the Hugging Face repository is private
hf download bowen12355/HARP-RPA \
  --local-dir weights \
  --exclude README.md \
  --exclude .gitattributes
```

After downloading, the `weights/` directory should contain:

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

To verify file integrity:

```bash
cd weights
sha256sum -c sha256sums.txt
```

Model checkpoints and pretrained weights are not tracked by git. The `weights/` directory in the GitHub repository only contains a README placeholder.

## Model Type

All HARP-RPA model configs use:

```yaml
model:
  model_type: harp_rpa
```

The model registry is defined in `models/model.py` through:

```python
@R.register("harp_rpa")
```

Use the current HARP-RPA configs listed below.

## Key Configurations

Available model configs:

```text
config/models/harp_rpa.yml
config/models/esmc_hpra_frozen.yml
config/models/esmc_rinalmo_seq.yml
config/models/ablate_ft_ddft.yml
config/models/ablate_ft_ddft_large.yml
config/models/ablate_ft_frozen_lm.yml
config/models/ablate_ft_lora_only.yml
```

Recommended main affinity model:

```bash
--model_config ./config/models/harp_rpa.yml
```

Sequence-only language-model baseline:

```bash
--model_config ./config/models/esmc_rinalmo_seq.yml
```

Fine-tuning ablations:

```text
ablate_ft_frozen_lm.yml    frozen language models
ablate_ft_lora_only.yml    LoRA-only adaptation
ablate_ft_ddft.yml         dual-depth fine-tuning
ablate_ft_ddft_large.yml   stronger top-layer fine-tuning
```

## Datasets

Expected dataset configs:

```text
config/datasets/PRA310.yml
config/datasets/PRA201.yml
config/datasets/mCSM.yml
config/datasets/blindtest.yml
```

PRA310 and PRA201 are used for supervised binding affinity prediction. The target is binding free energy `dG`. Mutation-effect experiments use mCSM-style `ddG` data when available.

Dataset files and structures should be placed under `./datasets` according to the paths in each dataset YAML file.

## Training

Run HARP-RPA on PRA310:

```bash
python run.py \
  --model_config ./config/models/harp_rpa.yml \
  --data_config ./config/datasets/PRA310.yml \
  --run_config ./config/runs/finetune_struct.yml \
  finetune dG
```

Run HARP-RPA on PRA201:

```bash
python run.py \
  --model_config ./config/models/harp_rpa.yml \
  --data_config ./config/datasets/PRA201.yml \
  --run_config ./config/runs/finetune_struct.yml \
  finetune dG
```

Run a sequence-only baseline:

```bash
python run.py \
  --model_config ./config/models/esmc_rinalmo_seq.yml \
  --data_config ./config/datasets/PRA310.yml \
  --run_config ./config/runs/finetune_struct.yml \
  finetune dG
```

## Testing

Evaluate a trained checkpoint on an affinity dataset:

```bash
python run.py \
  --model_config ./config/models/harp_rpa.yml \
  --data_config ./config/datasets/PRA310.yml \
  --run_config ./config/runs/test_basic.yml \
  test dG
```

For mutation-effect evaluation:

```bash
python run.py \
  --model_config ./config/models/harp_rpa.yml \
  --data_config ./config/datasets/mCSM.yml \
  --run_config ./config/runs/zero_shot_blindtest.yml \
  test ddG
```

## Training Notes

- The first epoch can be slower because structure features are cached.
- Main experiments should keep `pooling: token_interface`.
- Main and ablation experiments should use the same split, seed, optimizer, scheduler, and gradient clipping settings.
- Recommended gradient clipping for the current model is `max_grad_norm: 10.0`.
- If using ESMC, keep `esm_type: 'esmc-600m'`. Legacy ESM-2 branches are kept for compatibility with older experiments, but the paper model uses ESMC.

## Main Architecture

HARP-RPA follows this pipeline:

1. Encode protein and RNA sequences with ESMC and RiNALMo.
2. Assemble protein, RNA, and complex-level tokens.
3. Construct heterogeneous residue/base pair features from residue identity, relative index, chain assignment, coordinates, molecule type, valid atoms, and relation IDs.
4. Update token and pair representations with HeteroPairFormer.
5. Predict binding free energy with an interface-aware readout.

The relation IDs used by the heterogeneous pair encoder are:

```text
0: P-P same-chain
1: P-P inter-chain
2: R-R same-chain
3: R-R inter-chain
4: P-R cross-pair
5: complex-token relation
6: protein-token relation
7: RNA-token relation
```

## Output

Training outputs are written to the directory specified by the run config, for example:

```yaml
output_dir: ./outputs/esmc_rinalmo_ddft/...
```

Each fold writes logs, checkpoints, predictions, and metric CSV files under its corresponding output subdirectory.

## License

Original HARP-RPA code and modifications by Bowen Wang are
licensed under the [MIT License](LICENSE).

Third-party code retains its original copyright notices and licenses. In
particular, the bundled RiNALMo code remains under its Apache License 2.0
([RiNALMo/LICENSE](RiNALMo/LICENSE)). This project license does not replace
the licenses of third-party code, pretrained models, model weights, or datasets.
