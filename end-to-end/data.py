import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from datasets import load_dataset


# TODO: Change this to tmatch your dataset
class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove the batch dimension which tokenizer adds
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


# TODO: Change this logic to match your dataloading process.
# Notice the use of DistributedSampler for multi-gpu case.
def load_data(tokenizer, cfg, val_only=False):
    """Load and prepare the dataset."""

    if val_only:
        dataset = load_dataset("imdb", split="test")
        val_texts = dataset["text"][: cfg.data.test_size]
        val_labels = dataset["label"][: cfg.data.test_size]
    else:
        dataset = load_dataset("imdb")
        train_texts = (
            dataset["train"]["text"][: cfg.training.max_samples]
            if cfg.training.max_samples > 0
            else dataset["train"]["text"]
        )
        train_labels = (
            dataset["train"]["label"][: cfg.training.max_samples]
            if cfg.training.max_samples > 0
            else dataset["train"]["label"]
        )

        val_texts = (
            dataset["test"]["text"][: cfg.training.max_samples // 10]
            if cfg.training.max_samples > 0
            else dataset["test"]["text"][:5000]
        )
        val_labels = (
            dataset["test"]["label"][: cfg.training.max_samples // 10]
            if cfg.training.max_samples > 0
            else dataset["test"]["label"][:5000]
        )

        train_dataset = TextClassificationDataset(
            train_texts, train_labels, tokenizer, cfg.model.max_length
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=cfg.distributed.world_size,
            rank=cfg.distributed.rank,
            shuffle=True,
            seed=cfg.seed,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=train_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )

    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, cfg.model.max_length
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=cfg.distributed.world_size,
        rank=cfg.distributed.rank,
        shuffle=False,
        seed=cfg.seed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,
        sampler=val_sampler,
        num_workers=cfg.eval.num_workers,
        pin_memory=True,
    )

    if val_only:
        return val_loader
    else:
        return train_loader, val_loader, train_sampler
