import os
import torch
import torch.distributed as dist


def ddp_setup(cfg):
    """Initialize the distributed environment."""

    if "LOCAL_RANK" in os.environ:
        cfg.distributed.local_rank = int(os.environ["LOCAL_RANK"])
    if "WORLD_SIZE" in os.environ:
        cfg.distributed.world_size = int(os.environ["WORLD_SIZE"])
    if "RANK" in os.environ:
        cfg.distributed.rank = int(os.environ["RANK"])

    dist.init_process_group(backend=cfg.distributed.backend)
    torch.cuda.set_device(cfg.distributed.local_rank)
    print(
        f"Initialized process {cfg.distributed.rank}/{cfg.distributed.world_size} (local_rank: {cfg.distributed.local_rank})"
    )
    return cfg


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()
