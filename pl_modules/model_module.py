import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from models import ModelRegister
from utils.metrics import ScalarMetricAccumulator, cal_pearson, cal_spearman, cal_rmse, cal_mae, get_loss


def get_model(model_args: dict = None):
    register = ModelRegister()
    model_args_ori = {}
    model_args_ori.update(model_args)
    model_cls = register[model_args['model_type']]
    model = model_cls(**model_args_ori)
    return model


class ModelModule(pl.LightningModule):
    def __init__(self, output_dir=None, model_args=None, data_args=None, run_args=None):
        super().__init__()
        self.save_hyperparameters()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.l_type = data_args.loss_type
        self.model = get_model(model_args=model_args.model)
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.optimizers_cfg = self.model_args.train.optimizer
        self.scheduler_cfg = self.model_args.train.scheduler
        self.valid_it = 0
        self.batch_size = data_args.batch_size

        self.train_loss = None
        self.label_standardize = bool(getattr(self.model_args.train, 'label_standardize', False))
        self.label_std_eps = float(getattr(self.model_args.train, 'label_std_eps', 1e-6))
        label_mean, label_std = self._compute_label_stats()
        self.register_buffer('label_mean', torch.tensor(label_mean, dtype=torch.float32), persistent=True)
        self.register_buffer('label_std', torch.tensor(label_std, dtype=torch.float32), persistent=True)

    def _get_fold_col(self):
        fold_col = getattr(self.model_args, 'fold_col', None)
        if fold_col is not None:
            return fold_col
        if self.output_dir is not None:
            for part in Path(self.output_dir).parts:
                if part.startswith('log_fold_'):
                    return part.replace('log_', '')
        return 'fold_0'

    def _compute_label_stats(self):
        if not self.label_standardize:
            return 0.0, 1.0
        fold_col = self._get_fold_col()
        df = pd.read_csv(self.data_args.df_path)
        if fold_col not in df.columns:
            raise KeyError(f"Fold column {fold_col} not found in {self.data_args.df_path}")
        if self.data_args.col_label not in df.columns:
            raise KeyError(f"Label column {self.data_args.col_label} not found in {self.data_args.df_path}")
        train_labels = pd.to_numeric(
            df.loc[df[fold_col].isin(['train']), self.data_args.col_label],
            errors='coerce',
        ).dropna()
        if train_labels.empty:
            raise ValueError(f"No train labels found for {fold_col} in {self.data_args.df_path}")
        label_mean = float(train_labels.mean())
        label_std = float(train_labels.std(ddof=0))
        if not np.isfinite(label_std) or label_std < self.label_std_eps:
            label_std = 1.0
        print(
            f"[LabelStd] {fold_col}: mean={label_mean:.6f}, std={label_std:.6f}, "
            f"n_train={len(train_labels)}"
        )
        return label_mean, label_std

    def _standardize_labels(self, y):
        if not self.label_standardize:
            return y
        return (y - self.label_mean.to(y.device)) / self.label_std.to(y.device).clamp_min(self.label_std_eps)

    def _restore_labels(self, y):
        if not self.label_standardize:
            return y
        return y * self.label_std.to(y.device) + self.label_mean.to(y.device)

    def _build_optimizer_params(self):
        trunk_lr = getattr(self.optimizers_cfg, 'trunk_lr', None)
        if trunk_lr is None:
            return self.parameters()

        trunk_prefixes = (
            'model.pair_encoder',
            'model.c_former',
            'model.interface_pde',
            'model.pred_head',
            'model.ddg_pair_gate',
            'model.ddg_head',
            'model.z_proj',
            'model.proj_cplx',
            'model.project_feat',
            'model.parallel_gat',
            'model.complex_embedding',
            'model.prot_embedding',
            'model.rna_embedding',
        )
        base_params = []
        trunk_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(trunk_prefixes):
                trunk_params.append(param)
            else:
                base_params.append(param)

        print(
            "[OPT] param groups -> "
            f"base_lr={self.optimizers_cfg.lr}, base_params={sum(p.numel() for p in base_params) / 1e6:.2f}M | "
            f"trunk_lr={trunk_lr}, trunk_params={sum(p.numel() for p in trunk_params) / 1e6:.2f}M"
        )
        return [
            {'params': base_params, 'lr': self.optimizers_cfg.lr},
            {'params': trunk_params, 'lr': trunk_lr},
        ]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        if self.optimizers_cfg.type == 'adam':
            optimizer = torch.optim.Adam(
                self._build_optimizer_params(),
                lr=self.optimizers_cfg.lr,
                betas=(self.optimizers_cfg.beta1, self.optimizers_cfg.beta2),
            )
        elif self.optimizers_cfg.type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizers_cfg.lr)
        elif self.optimizers_cfg.type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizers_cfg.lr)
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizers_cfg.type)

        if self.scheduler_cfg.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.scheduler_cfg.factor,
                patience=self.scheduler_cfg.patience,
                min_lr=self.scheduler_cfg.min_lr,
            )
        elif self.scheduler_cfg.type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.scheduler_cfg.milestones,
                gamma=self.scheduler_cfg.gamma,
            )
        elif self.scheduler_cfg.type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.scheduler_cfg.gamma,
            )
        else:
            raise NotImplementedError('Scheduler not supported: %s' % self.scheduler_cfg.type)

        if self.model_args.resume is not None:
            print("Resuming from checkloint: %s" % self.model_args.resume)
            ckpt = torch.load(self.model_args.resume, map_location=self.model_args.device)
            it_first = ckpt['iteration']
            lsd_result = self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
            print('Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])

        # Keep the scheduler on val_loss for stability.
        if self.scheduler_cfg.type == 'plateau':
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss',
                },
            }
        else:
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        return optim_dict

    def on_train_start(self):
        log_hyperparams = {'model_args': self.model_args, 'data_args': self.data_args, 'run_args': self.run_args}
        self.logger.log_hyperparams(log_hyperparams)

    def on_before_optimizer_step(self, optimizer) -> None:
        pass

    def training_step(self, batch, batch_idx):
        y_raw = batch['labels']
        y = self._standardize_labels(y_raw)
        pred = self.model(batch, self.data_args.strategy)
        loss = get_loss(self.l_type, pred, y, reduction='mean')
        self.train_loss = loss.detach()
        self.log(
            "train_loss",
            float(self.train_loss),
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        y_raw = batch['labels']
        y = self._standardize_labels(y_raw)
        pred = self.model(batch, self.data_args.strategy)
        val_loss = get_loss(self.l_type, pred, y, reduction='mean')
        pred_raw = self._restore_labels(pred).detach()
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=self.batch_size, mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for y_true, y_pred in zip(y_raw, pred_raw):
            result = {
                'y_true': y_true.item(),
                'y_pred': y_pred.item(),
            }
            self.results.append(result)
        return val_loss

    def on_validation_epoch_end(self):
        results = pd.DataFrame(self.results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)

        y_pred = np.array(results['y_pred'])
        y_true = np.array(results['y_true'])

        pearson_all = np.abs(cal_pearson(y_pred, y_true))
        spearman_all = np.abs(cal_spearman(y_pred, y_true))
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')

        self.log('val/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log('val/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True, prog_bar=False)

        val_loss = rmse_all * rmse_all
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True, prog_bar=False)

        # Composite metric: higher is better.
        val_composite = (
            0.40 * pearson_all
            + 0.30 * spearman_all
            + 0.15 / (1.0 + rmse_all)
            + 0.15 / (1.0 + mae_all)
        )
        self.log('val_composite', val_composite, batch_size=self.batch_size, on_epoch=True, sync_dist=True, prog_bar=True)

        self.valid_it += 1
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.results = []
        self.scalar_accum = ScalarMetricAccumulator()

    def test_step(self, batch, batch_idx):
        y_raw = batch['labels']
        y = self._standardize_labels(y_raw)
        pred = self.model(batch, self.data_args.strategy)
        test_loss = get_loss(self.l_type, pred, y, reduction='mean')
        pred_raw = self._restore_labels(pred).detach()
        self.scalar_accum.add(name='loss', value=test_loss, batchsize=self.batch_size, mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for y_true, y_pred in zip(y_raw, pred_raw):
            result = {
                'y_true': y_true.item(),
                'y_pred': y_pred.item(),
            }
            self.results.append(result)
        return test_loss

    def on_test_epoch_end(self):
        results = pd.DataFrame(self.results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_test.csv'), index=False)
        y_pred = np.array(results['y_pred'])
        y_true = np.array(results['y_true'])
        pearson_all = np.abs(cal_pearson(y_pred, y_true))
        spearman_all = np.abs(cal_spearman(y_pred, y_true))
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        # print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')

        self.log('test/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log('test/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log('test/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log('test/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.res = {"pearson": pearson_all, "spearman": spearman_all, "rmse": rmse_all, "mae": mae_all}
        print("Self.Res:", self.res)

        test_loss = rmse_all * rmse_all
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        return test_loss
