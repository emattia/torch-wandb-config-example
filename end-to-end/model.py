from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import BertModel, BertTokenizer
import wandb


# TODO: Change this to the nn.Module(s) you want to train
class BertClassifier(nn.Module):
    """BERT-based text classifier."""

    def __init__(self, num_classes, pretrained_model="bert-base-uncased"):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# TODO: This is relevant if you are building a language model that needs to tokenize text.
def load_tokenizer(cfg):
    return BertTokenizer.from_pretrained(cfg.model.model_name)


# TODO: Change this to update the training loop with any custom training/logging logic.
def train_epoch(model, train_loader, optimizer, criterion, epoch, cfg):
    """Train for one epoch."""
    model.train()
    train_sampler = train_loader.sampler
    train_sampler.set_epoch(
        epoch
    )  # Important for proper shuffling in distributed training

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = None
    if cfg.distributed.rank == 0 and not cfg.metaflow.use_metaflow:
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")

    ### MAIN TRAINING LOGIC FOR ONE EPOCH ###
    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # TODO: Set up any per batch logging.
        if cfg.wandb.use_wandb and cfg.distributed.rank == 0 and i % 50 == 0:
            step = epoch * len(train_loader) + i
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_accuracy": 100
                    * (predicted == labels).sum().item()
                    / labels.size(0),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "step": step,
                }
            )

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "loss": total_loss / (progress_bar.n + 1),
                    "acc": 100 * correct / total,
                }
            )

    if progress_bar is not None:
        progress_bar.close()

    ### GATHER METRICS ###
    metrics = torch.tensor(
        [total_loss, correct, total], dtype=torch.float32, device="cuda"
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    # TODO: Set up any per epoch logging.
    avg_loss = metrics[0].item() / metrics[2].item()
    accuracy = 100 * metrics[1].item() / metrics[2].item()
    if cfg.wandb.use_wandb and cfg.distributed.rank == 0:
        wandb.log(
            {
                "train/epoch": epoch,
                "train/loss": avg_loss,
                "train/accuracy": accuracy,
                "epoch": epoch,
            }
        )

    return avg_loss, accuracy


# The reason for this abstraction is checkpoint saving can work out of the box with just
# `model.state_dict()` since the model is wrapped in `DDP` and when checkpoints are loaded
# the model is already wrapped in the DPP wrapper. But when models are loaded outside the training
# context (i.e. outside the distributed wrapper), we need to ensure that the model's original state
# is loaded and not the `DDP` wrapped model.
def save_model(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            # When we save model we need to ensure that it's stripping the `DDP` wrapper
            # this because the DDP wrapper will prefix the `module` to the model's state_dict
            # and when we load the model, we need to ensure that we are loading the model
            # in a way that works outside the distributed context
            "model_state_dict": model.module.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_model_from_checkpoint(checkpoint_path, cfg):
    """Load a trained model from checkpoint."""

    model = BertClassifier(
        num_classes=cfg.model.num_classes, pretrained_model=cfg.model.model_name
    )
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Load state dict into model
    model.load_state_dict(state_dict)
    model = model.cuda()

    # Wrap in DDP if distributed
    if cfg.distributed.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.distributed.local_rank],
            output_device=cfg.distributed.local_rank,
        )

    return model
