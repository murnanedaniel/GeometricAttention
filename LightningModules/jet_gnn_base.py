import sys, os
import logging
import time
import warnings

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .utils import load_processed_datasets

class JetGNNBase(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        
        """
        Initialise the Lightning Module that can scan over different Equivariant GNN training regimes
        """
        # Assign hyperparameters
        self.save_hyperparameters(hparams)

        if "graph_construction" not in self.hparams: self.hparams["graph_construction"] = None
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage="fit"):

        data_split = self.hparams["data_split"]

        if stage == "fit":
            data_split[2] = 0 # No test set in training
        elif stage == "test":
            data_split[0], data_split[1] = 0, 0 # No train or val set in testing

        if (self.trainset is None) and (self.valset is None) and (self.testset is None):
            self.trainset, self.valset, self.testset = load_processed_datasets(self.hparams["input_dir"], 
                                                        data_split,
                                                        self.hparams["graph_construction"]
                                                        )
        
        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
            self.logger.experiment.define_metric("acc" , summary="max")
            self.logger.experiment.define_metric("inv_eps" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")
            
        print(time.ctime())
        
    def concat_feature_set(self, batch):
        """
        Useful in all models to use all available features of size == len(x)
        """
        
        all_features = []
        for feature in self.hparams["feature_set"]:
            if len(batch[feature]) == len(batch.x):
                all_features.append(batch[feature])
            else:
                all_features.append(batch[feature][batch.batch])
        return torch.stack(all_features).T

    def get_metrics(self, targets, output):
        
        prediction = torch.sigmoid(output)
        tp = (prediction.round() == targets).sum().item()
        acc = tp / len(targets)
        
        try:
            auc = roc_auc_score(targets.bool().cpu().detach(), prediction.cpu().detach())
        except Exception:
            auc = 0
        fpr, tpr, _ = roc_curve(targets.bool().cpu().detach(), prediction.cpu().detach())
        
        # Calculate which threshold gives the best signal goal
        signal_goal_idx = abs(tpr - self.hparams["signal_goal"]).argmin()
        
        eps_fpr = fpr[signal_goal_idx]
        eps_tpr = tpr[signal_goal_idx]

        
        return acc, auc, eps_fpr, eps_tpr
    
    def apply_loss_function(self, output, batch):
        return F.binary_cross_entropy_with_logits(output, batch.y.float(), pos_weight=torch.tensor(self.hparams["pos_weight"]))

    def training_step(self, batch, batch_idx):
                
        output = self(batch).squeeze(-1)

        loss = self.apply_loss_function(output, batch)
        
        acc, auc, eps, eps_eff = self.get_metrics(batch.y.bool(), output)
        
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True)

        return loss        

    def shared_val_step(self, batch):

        output = self(batch).squeeze(-1)

        loss = self.apply_loss_function(output, batch)

        acc, auc, eps, eps_eff = self.get_metrics(batch.y.bool(), output)
        
        current_lr = self.optimizers().param_groups[0]["lr"]
        
        self.log_dict({"val_loss": loss, "current_lr": current_lr}, on_step=False, on_epoch=True)
        
        return {
            "loss": loss,
            "outputs": output,
            "targets": batch.y,
            "acc": acc,
            "auc": auc,
            "eps": eps,
            "eps_eff": eps_eff,
        }

    def validation_step(self, batch, batch_idx):
        return self.shared_val_step(batch)

    def test_step(self, batch, batch_idx):
        return self.shared_val_step(batch)
        
    def shared_end_step(self, step_outputs):
        # Concatenate all predictions and targets
        preds = torch.cat([output["outputs"] for output in step_outputs])
        targets = torch.cat([output["targets"] for output in step_outputs])

        # Calculate the ROC curve
        acc, auc, eps, eps_eff = self.get_metrics(targets, preds)

        if eps != 0:
            self.log_dict({"acc": acc, "auc": auc, "inv_eps": 1/eps, "eps_eff": eps_eff})
    
    def validation_epoch_end(self, step_outputs):
        self.shared_end_step(step_outputs)

    def test_epoch_end(self, step_outputs):
        self.shared_end_step(step_outputs)

        
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["train_batch"], num_workers=1, shuffle=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["val_batch"], num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0001,
                amsgrad=True,
            )
        ]
        if "scheduler" not in self.hparams or self.hparams["scheduler"] is None or self.hparams["scheduler"] == "StepLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.StepLR(
                        optimizer[0],
                        step_size=self.hparams["patience"],
                        gamma=self.hparams["factor"],
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        elif self.hparams["scheduler"] == "CosineWarmLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer[0],
                        T_0 = self.hparams["patience"], 
                        T_mult=2,
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
        return optimizer, scheduler
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        