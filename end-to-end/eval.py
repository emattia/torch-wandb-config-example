#!/usr/bin/env python3
"""
Test script for BERT classifier model trained with train.py.
Can be used standalone or called from Metaflow workflow.

Check out the TODOs, and adapt to your dataset/module/hyperparams/etc.
"""

import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from omegaconf import DictConfig, OmegaConf
import hydra
from dist_utils import ddp_setup, cleanup
from data import load_data
from model import load_model_from_checkpoint, load_tokenizer


def validate(model, val_loader, criterion, cfg, epoch):
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ### Forward pass over validation DataLoader.
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    metrics = torch.tensor(
        [total_loss, correct, total], dtype=torch.float32, device="cuda"
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    avg_loss = metrics[0].item() / len(val_loader) / cfg.distributed.world_size
    accuracy = 100 * metrics[1].item() / metrics[2].item()

    # TODO: Set up any custom evaluation logging.
    if cfg.wandb.use_wandb and cfg.distributed.rank == 0:
        wandb.log({"val/loss": avg_loss, "val/accuracy": accuracy, "epoch": epoch})

    return avg_loss, accuracy


def detailed_evaluation(model, test_loader, criterion, cfg):
    """Perform detailed evaluation with additional metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    # Get local rank for distributed processing
    rank = cfg.distributed.rank
    world_size = cfg.distributed.world_size
    is_distributed = world_size > 1 and torch.distributed.is_initialized()

    # Evaluate model on test data
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", disable=rank != 0):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays (local to this process)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Gather results from all processes if in distributed mode
    if is_distributed:
        # First, count how many items each process has
        local_count = torch.tensor([len(all_preds)], device="cuda")
        all_counts = [torch.zeros(1, device="cuda") for _ in range(world_size)]
        torch.distributed.all_gather(all_counts, local_count)

        # Convert predictions and labels to tensors
        pred_tensor = torch.tensor(all_preds, device="cuda")
        label_tensor = torch.tensor(all_labels, device="cuda")

        # Gather loss values for averaging
        loss_tensor = torch.tensor([total_loss], device="cuda")
        all_losses = [torch.zeros(1, device="cuda") for _ in range(world_size)]
        torch.distributed.all_gather(all_losses, loss_tensor)

        # Only process 0 will combine and analyze results
        if rank == 0:
            # Sum of all losses
            total_loss = sum(loss.item() for loss in all_losses)

            # Create storage for all predictions and labels
            gathered_preds = []
            gathered_labels = []

            # Add this process's data first
            gathered_preds.append(all_preds)
            gathered_labels.append(all_labels)

            # Create padded tensors for each rank's data
            for r in range(1, world_size):  # Skip rank 0 (already added)
                rank_count = int(all_counts[r].item())

                # Skip if this rank has no data
                if rank_count == 0:
                    continue

                # Create tensors to receive data
                r_preds = torch.zeros(rank_count, dtype=torch.long, device="cuda")
                r_labels = torch.zeros(rank_count, dtype=torch.long, device="cuda")

                # Gather from other ranks
                # Note: We need to use point-to-point communication here because all_gather
                # requires tensors of the same size on all processes

                # First send from other ranks to rank 0
                if rank == 0:
                    # Receive data from rank r
                    torch.distributed.recv(r_preds, src=r)
                    torch.distributed.recv(r_labels, src=r)

                    # Add to gathered data
                    gathered_preds.append(r_preds.cpu().numpy())
                    gathered_labels.append(r_labels.cpu().numpy())

            # Concatenate all data
            all_preds = np.concatenate(gathered_preds)
            all_labels = np.concatenate(gathered_labels)

        # Other ranks send their data to rank 0
        else:
            if local_count.item() > 0:  # Only send if we have data
                torch.distributed.send(pred_tensor, dst=0)
                torch.distributed.send(label_tensor, dst=0)

    # Only rank 0 processes the results to avoid redundant computation
    if rank == 0:
        # Get the number of samples
        n_samples = len(all_preds)

        # Average the loss
        avg_loss = total_loss / len(test_loader) / (world_size if is_distributed else 1)

        # Calculate accuracy
        accuracy = np.mean(all_preds == all_labels) * 100

        # Identify which classes are actually present in the data
        unique_labels_in_data = sorted(
            np.unique(np.concatenate([all_labels, all_preds]))
        )

        # Get expected class names from config
        class_names = (
            cfg.model.class_names if hasattr(cfg.model, "class_names") else None
        )

        # Check if class names match the classes in the data
        valid_class_names = class_names is not None and len(
            unique_labels_in_data
        ) == len(class_names)

        # Calculate confusion matrix for all classes (including those not in the data)
        expected_classes = list(range(cfg.model.num_classes))
        cm = confusion_matrix(
            all_labels,
            all_preds,
            labels=expected_classes,  # Always use all possible classes
        )

        # Try to generate a detailed classification report
        try:
            if valid_class_names:
                # Use provided class names if they match the data
                report = classification_report(
                    all_labels,
                    all_preds,
                    target_names=class_names,
                    labels=expected_classes,  # Ensure we use all expected classes
                    output_dict=True,
                    zero_division=0,  # Handle classes with no samples
                )
            else:
                # Fall back to numeric class labels
                report = classification_report(
                    all_labels,
                    all_preds,
                    labels=expected_classes,
                    output_dict=True,
                    zero_division=0,
                )
        except ValueError as e:
            print(f"Warning: Could not generate detailed classification report: {e}")
            print(f"Found classes in data: {unique_labels_in_data}")
            print(f"Expected classes: {expected_classes}")

            # Create a simplified report
            report = {
                "accuracy": accuracy,
                "found_classes": [int(l) for l in unique_labels_in_data],
                "expected_classes": expected_classes,
            }

            # Add per-class metrics manually for classes that are present
            for label in unique_labels_in_data:
                # For classes that are present, calculate basic metrics
                class_mask = all_labels == label
                if np.sum(class_mask) > 0:  # If we have samples for this class
                    class_pred = all_preds == label
                    true_pos = np.sum(class_pred & class_mask)
                    false_pos = np.sum(class_pred & ~class_mask)
                    false_neg = np.sum(~class_pred & class_mask)

                    # Calculate metrics (with protection against division by zero)
                    precision = true_pos / max(true_pos + false_pos, 1)
                    recall = true_pos / max(true_pos + false_neg, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

                    class_name = class_names[label] if valid_class_names else str(label)
                    report[class_name] = {
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1,
                        "support": np.sum(class_mask),
                    }

        # Combine all results
        results = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "num_samples": n_samples,
        }

        return results

    return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@hydra.main(version_base=None, config_path="conf", config_name="pytorch_eval_config")
def main(cfg: DictConfig):
    """Main function for model evaluation."""

    if cfg.use_distributed:
        cfg = ddp_setup(cfg)
    else:
        if (
            "LOCAL_RANK" in os.environ
            or "WORLD_SIZE" in os.environ
            or "RANK" in os.environ
        ) and (int(os.environ["WORLD_SIZE"]) > 1):
            raise ValueError(
                "Detected torchrun environment variables (LOCAL_RANK, WORLD_SIZE, or RANK) "
                "and WORLD_SIZE is > 1 "
                "but 'use_distributed' is set to False in config. "
            )

        if (
            "LOCAL_RANK" in os.environ
            or "WORLD_SIZE" in os.environ
            or "RANK" in os.environ
        ):
            cfg.use_distributed = (
                True  # User ran via torchrun, so do this to clean process groups.
            )
            dist.init_process_group(backend=cfg.distributed.backend)

        cfg.distributed.rank = 0
        cfg.distributed.world_size = 1
        cfg.distributed.local_rank = 0
        torch.cuda.set_device(0)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.distributed.rank == 0:
        print("\n---------------")
        print("Evaluation config")
        print("---------------\n")
        print(OmegaConf.to_yaml(cfg))
        print(f"\nEvaluating model from: {cfg.model_path}")
        print(f"Using {cfg.distributed.world_size} GPU(s)\n")

    tokenizer = load_tokenizer(cfg)
    model = load_model_from_checkpoint(cfg.model_path, cfg)
    model.eval()
    test_loader = load_data(tokenizer, cfg, val_only=True)
    criterion = torch.nn.CrossEntropyLoss()

        # TODO: Log any parameters you'd like to see in wandb.
    if cfg.wandb.use_wandb and cfg.distributed.rank == 0:
        wandb_config = {
            "model_name": cfg.model.model_name,
            "batch_size": cfg.eval.batch_size,
            "num_gpus": cfg.distributed.world_size,
            "seed": cfg.seed,
        }

        run_name = (
            cfg.wandb.name
            if cfg.wandb.name
            else f"bert-{cfg.model.model_name.split('/')[-1]}-bs{cfg.eval.batch_size}"
        )
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=wandb_config,
        )

        wandb.watch(model, log="all", log_freq=100)

    val_loss, val_acc = validate(model, test_loader, criterion, cfg, epoch=0)

    if cfg.distributed.rank == 0:
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    if cfg.detailed_evaluation:
        results = detailed_evaluation(model, test_loader, criterion, cfg)

        if cfg.distributed.rank == 0:
            print("\nDetailed Evaluation Results:")
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Loss: {results['loss']:.4f}")

            report_df = pd.DataFrame(results["classification_report"]).transpose()
            print("\nClassification Report:")
            print(report_df)

            print("\nConfusion Matrix:")
            cm = np.array(results["confusion_matrix"])
            print(cm)

            if cfg.output_file:
                with open(cfg.output_file, "w") as f:
                    json.dump(results, f, indent=4, cls=NumpyEncoder)
                print(f"\nResults saved to: {cfg.output_file}")

    if cfg.use_distributed:
        cleanup()

    if cfg.distributed.rank == 0:
        print("Evaluation completed!")


if __name__ == "__main__":
    main()
