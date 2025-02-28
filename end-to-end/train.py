#!/usr/bin/env python3
"""
Train script for BERT classifier model trained with train.py.
Can be used standalone or called from Metaflow workflow.

Check out the TODOs, and adapt to your dataset/module/hyperparams/etc.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
from metaflow import Checkpoint

### TODO: Investigate each module.
# Understand and customize to your liking.
# This is where to put your business logic.
from dist_utils import ddp_setup, cleanup
from data import load_data
from model import BertClassifier, load_tokenizer, train_epoch, save_model
from eval import validate


# The main loop that is the entrypoint of torchrun.
@hydra.main(version_base=None, config_path="conf", config_name="pytorch_train_config")
def main(cfg: DictConfig):

    cfg = ddp_setup(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.distributed.local_rank == 0:
        print("\n---------------")
        print("Training config")
        print("---------------\n")
        print(OmegaConf.to_yaml(cfg))

    if cfg.distributed.rank == 0:
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        print(f"Training BERT on {cfg.distributed.world_size} GPUs")

    # TODO: Replace with the module you want to optimize.
    tokenizer = load_tokenizer(cfg)
    model = BertClassifier(
        num_classes=cfg.model.num_classes, pretrained_model=cfg.model.model_name
    )
    model = model.cuda()
    model = DDP(
        model,
        device_ids=[cfg.distributed.local_rank],
        output_device=cfg.distributed.local_rank,
        find_unused_parameters=False,
    )

    # TODO: Replace with your loss function and optimization strategy.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # TODO: Update in accord with how your load_data function is working.
    train_loader, val_loader, _ = load_data(tokenizer, cfg)

    # TODO: Log any parameters you'd like to see in wandb.
    if cfg.wandb.use_wandb and cfg.distributed.rank == 0:
        wandb_config = {
            "model_name": cfg.model.model_name,
            "batch_size": cfg.training.batch_size
            * cfg.distributed.world_size,  # Global batch size
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "epochs": cfg.training.epochs,
            "max_length": cfg.model.max_length,
            "num_gpus": cfg.distributed.world_size,
            "seed": cfg.seed,
        }

        run_name = (
            cfg.wandb.name
            if cfg.wandb.name
            else f"bert-{cfg.model.model_name.split('/')[-1]}-bs{cfg.training.batch_size}-lr{cfg.training.learning_rate}"
        )
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=wandb_config,
        )

        wandb.watch(model, log="all", log_freq=100)

    if cfg.metaflow.checkpoint_in_remote_datastore:
        checkpoint = Checkpoint(init_dir=True)
        best_path = os.path.join(checkpoint.directory, "best_model.pth")
        epoch_path = os.path.join(checkpoint.directory, "epoch.pth")
    else:
        best_path = os.path.join(cfg.training.output_dir, "best_model.pth")
        epoch_path = os.path.join(cfg.training.output_dir, "epoch.pth")
    best_val_loss = float("inf")

    ### MAIN TRAINING LOOP LOGIC ###
    for epoch in range(cfg.training.epochs):
        if cfg.distributed.rank == 0:
            print(f"\nEpoch {epoch+1}/{cfg.training.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, epoch, cfg
        )
        val_loss, val_acc = validate(model, val_loader, criterion, cfg, epoch)

        if cfg.distributed.rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        ### SAVE CHECKPOINT
        if (epoch + 1) % cfg.training.save_every == 0:

            if cfg.distributed.rank == 0:
                chckpt_metadata = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "model_name": cfg.model.model_name,
                    "batch_size": cfg.training.batch_size,
                    "learning_rate": cfg.training.learning_rate,
                }

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    epoch_path,
                )

                if cfg.metaflow.checkpoint_in_remote_datastore:
                    print("Saving epoch_checkpoint to persistent storage...")
                    checkpoint.save(
                        epoch_path,
                        metadata=chckpt_metadata,
                        name=f"epoch_checkpoint",
                        latest=not cfg.training.best_is_latest,
                    )
                    print("Done!")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": val_loss,
                        },
                        best_path,
                    )
                    if cfg.metaflow.checkpoint_in_remote_datastore:
                        print("Saving best_model checkpoint to persistent storage...")
                        checkpoint.save(
                            best_path,
                            metadata=chckpt_metadata,
                            name="best_model",
                            latest=cfg.training.best_is_latest,
                        )
                        print("Done!")

    if cfg.wandb.use_wandb and cfg.distributed.rank == 0:
        wandb.finish()

    if cfg.distributed.rank == 0 and cfg.metaflow.checkpoint_in_remote_datastore:
        if cfg.metaflow.final_model_path is not None:
            print(
                "Saving the model to",
                cfg.metaflow.final_model_path,
            )
            save_model(
                model, optimizer, cfg.training.epochs, cfg.metaflow.final_model_path
            )

    cleanup()
    if cfg.distributed.rank == 0:
        print("Training completed!")


if __name__ == "__main__":
    main()
