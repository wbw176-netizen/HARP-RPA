import json
import os
os.environ["NUMEXPR_MAX_THREADS"] = "56"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import time
from pathlib import Path
from collections import defaultdict

import fire
import pandas as pd
import numpy as np
import yaml
import wandb
from easydict import EasyDict

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.strategies.ddp import DDPStrategy

from pl_modules import ModelModule, DataModule, DDGModule


torch.set_num_threads(16)


def parse_yaml(yaml_dir):
    if not os.path.exists(yaml_dir):
        raise FileNotFoundError(f"Config file not found: {yaml_dir}")

    with open(yaml_dir, "r", encoding="utf-8") as f:
        content = f.read()
        config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
    return config_dict


def init_pytorch_settings():
    # Avoid repeatedly setting multiprocessing start method.
    try:
        torch.multiprocessing.set_start_method("forkserver", force=False)
    except RuntimeError:
        pass

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(4)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class LightningRunner(object):
    def __init__(
        self,
        model_config="./config/models/harp_rpa.yml",
        data_config="./config/datasets/PRA310.yml",
        run_config="./config/runs/test_basic.yml",
    ):
        super(LightningRunner, self).__init__()
        self.model_args = parse_yaml(model_config)
        self.dataset_args = parse_yaml(data_config)
        self.run_args = parse_yaml(run_config)
        init_pytorch_settings()

    def _get_module_class(self, stage):
        if stage == "dG":
            return ModelModule
        elif stage == "ddG":
            return DDGModule
        else:
            raise NotImplementedError(f"Unknown stage: {stage}")

    def _resolve_num_devices(self, gpus):
        if isinstance(gpus, int):
            return gpus
        elif isinstance(gpus, (list, tuple)):
            return len(gpus)
        else:
            return 1

    def _resolve_precision(self):
        if not torch.cuda.is_available():
            return 32
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        return "16-mixed"

    def _resolve_trainer_strategy(self, gpus):
        num_devices = self._resolve_num_devices(gpus)
        if torch.cuda.is_available() and num_devices > 1:
            return DDPStrategy(find_unused_parameters=True)
        return "auto"

    def _resolve_monitor_metric(self):
        monitor_metric = getattr(self.run_args, "monitor_metric", "val_composite")
        if monitor_metric == "val_loss":
            return monitor_metric, "min"
        if monitor_metric in {"val/all_rmse", "val/all_mae"}:
            return monitor_metric, "min"
        return monitor_metric, "max"

    def _build_callbacks(self, log_dir):
        monitor_metric, monitor_mode = self._resolve_monitor_metric()
        filename_metric = monitor_metric.replace("/", "_")

        callbacks = []
        callbacks.extend(
            [
                EarlyStopping(
                    monitor=monitor_metric,
                    mode=monitor_mode,
                    patience=self.run_args.patience,
                    strict=False,
                ),
                ModelCheckpoint(
                    dirpath=(log_dir / "checkpoint"),
                    filename="{epoch}-{" + filename_metric + ":.4f}",
                    monitor=monitor_metric,
                    mode=monitor_mode,
                    save_last=False,
                    save_top_k=1,
                    save_weights_only=False,
                    auto_insert_metric_name=False,
                ),
                TQDMProgressBar(refresh_rate=1),
            ]
        )
        return callbacks

    def save_model(self, output_dir, trainer, stage):
        best_ckpt = trainer.checkpoint_callback.best_model_path
        print("Best Model Path:", best_ckpt)

        module_cls = self._get_module_class(stage)
        module = module_cls.load_from_checkpoint(best_ckpt)

        if trainer.global_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)

            (output_dir / "model_data.json").write_text(
                json.dumps(vars(self.dataset_args), indent=2),
                encoding="utf-8",
            )


            torch.save(module.state_dict(), str(output_dir / "lightning_module_state_dict.pt"))



    def select_module(self, stage, log_dir):
        if stage == "dG":
            model = ModelModule(
                output_dir=log_dir,
                model_args=self.model_args,
                data_args=self.dataset_args,
                run_args=self.run_args,
            )
        elif stage == "ddG":
            model = DDGModule(
                output_dir=log_dir,
                model_args=self.model_args,
                data_args=self.dataset_args,
                run_args=self.run_args,
            )
        else:
            raise NotImplementedError
        return model

    def finetune(self, stage="dG"):
        print("Run args:", self.run_args, "\n")
        print("Model args:", self.model_args, "\n")
        print("Dataset args:", self.dataset_args, "\n")

        output_dir = Path(self.run_args.output_dir)
        gpus = self.run_args.gpus
        self.model_args.model.stage = stage

        run_name = self.run_args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
        run_results = []

        num_devices = self._resolve_num_devices(gpus)
        strategy = self._resolve_trainer_strategy(gpus)
        precision = self._resolve_precision()

        for k in range(self.run_args.num_folds):
            print(f"Training fold {k} Started!")

            log_dir = output_dir / f"log_fold_{k}"
            fold_col = f"fold_{k}"
            data_module = DataModule(
                dataset_args=self.dataset_args,
                **self.dataset_args,
                col_group=fold_col,
            )
            self.model_args.fold_col = fold_col

            model = self.select_module(stage, log_dir)

            if self.run_args.wandb:
                logger = WandbLogger(project="harp_rpa", name=f"{run_name}_fold{k}")
            else:
                logger = CSVLogger(str(log_dir))

            pl.seed_everything(self.model_args.train.seed, workers=True)
            print("Successfully initialized, start trainer...")

            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=gpus if torch.cuda.is_available() else 1,
                strategy=strategy,
                precision=precision,
                accumulate_grad_batches=getattr(self.run_args, "accumulate_grad_batches", 8),
                max_epochs=self.run_args.epochs,
                logger=logger,
                callbacks=self._build_callbacks(log_dir),
                gradient_clip_val=(
                    self.model_args.train.max_grad_norm
                    if getattr(self.model_args.train, "max_grad_norm", None) is not None
                    else None
                ),
                gradient_clip_algorithm=(
                    "norm"
                    if getattr(self.model_args.train, "max_grad_norm", None) is not None
                    else None
                ),
                log_every_n_steps=3,
            )

            trainer.fit(
                model=model,
                datamodule=data_module,
                ckpt_path=self.run_args.ckpt,
            )

            print(f"Training fold {k} Finished!")

            if num_devices > 1:
                trainer.strategy.barrier()

            print("Best Validation Results:")
            _ = trainer.test(model=model, ckpt_path="best", datamodule=data_module)

            res = model.res
            run_results.append(res)

            if trainer.global_rank == 0:
                self.save_model(log_dir, trainer, stage)

            if self.run_args.wandb:
                wandb.finish()

        result_dir = output_dir / run_name
        result_dir.mkdir(parents=True, exist_ok=True)

        with open(result_dir / "res.json", "w", encoding="utf-8") as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)

        results_df = pd.DataFrame(run_results)
        print(results_df.describe())

    def test(self, stage="dG"):
        print("Args:", self.run_args, self.dataset_args, self.model_args)

        output_dir = Path(self.run_args.output_dir)
        ckpts = self.run_args.ckpts
        gpus = self.run_args.gpus

        run_results = []

        num_devices = self._resolve_num_devices(gpus)
        strategy = self._resolve_trainer_strategy(gpus)
        precision = self._resolve_precision()

        for k in range(self.run_args.num_folds):
            log_dir = output_dir / f"log_fold_{k}"

            fold_col = f"fold_{k}"
            data_module = DataModule(
                dataset_args=self.dataset_args,
                **self.dataset_args,
                col_group=fold_col,
            )
            self.model_args.fold_col = fold_col

            model = self.select_module(stage, log_dir)
            logger = CSVLogger(str(log_dir))

            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=gpus if torch.cuda.is_available() else 1,
                strategy=strategy,
                precision=precision,
                max_epochs=0,
                logger=logger,
                callbacks=[
                    TQDMProgressBar(refresh_rate=1),
                ],
            )

            _ = trainer.test(model=model, ckpt_path=ckpts[k], datamodule=data_module)
            res = model.res
            run_results.append(res)

        if trainer.global_rank == 0:
            results_df = pd.DataFrame(run_results)
            print(results_df.describe())


if __name__ == "__main__":
    fire.Fire(LightningRunner)


