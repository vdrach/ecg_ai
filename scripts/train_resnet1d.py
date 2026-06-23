"""
train_resnet1d.py

DDP-aware training script for the parametric 1D ResNet. Single-node
multi-GPU via torchrun; manual GPU-resident batching; rank-aware data
sharding for strong scaling.

Usage (single GPU, no DDP):
    python train_resnet1d.py --base_width 64 --depth 2

Usage (multi-GPU on one node, e.g. 4 GPUs):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        train_resnet1d.py --base_width 64 --depth 2

When --result_csv is set, only rank 0 writes the CSV row. All ranks
share their epoch time via all_reduce(MAX) so the reported epoch time
is the slowest rank (true wall-clock).
"""

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import train_utils as tu

from pathlib import Path
import sys

project_root = Path.cwd()
while not (project_root / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
print(f"project root found: {project_root}")

def parse_args():
    parser = argparse.ArgumentParser()
    tu.add_common_args(parser)
    parser.add_argument("--base_width", type=int, default=64,
                        help="ResNet channel width at first stage; doubles each stage. "
                             "Roughly squares FLOPs when doubled.")
    parser.add_argument("--depth", type=int, default=2,
                        help="Residual blocks per stage. Linear in FLOPs/params.")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device = tu.setup_distributed()

    # TF32: global backend switch. Must happen before model construction.
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.batch_size % world_size != 0:
        if rank == 0:
            print(f"ERROR: --batch_size {args.batch_size} not divisible by "
                  f"world_size {world_size} -- adjust --batch_size", file=sys.stderr)
        tu.cleanup_distributed(world_size)
        sys.exit(1)
    per_rank_batch_size = args.batch_size // world_size

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    # Project setup -- needs to happen on every rank (each loads its
    # own copy of the dataset; we don't try to share across ranks since
    # the dataset fits easily in VRAM and copying-on-load is once-per-run).
    project_root = tu.find_project_root()
    sys.path.insert(0, str(project_root))

    if rank == 0:
        print(f"project root found: {project_root}")
        extra_lines = [
            ("architecture", "ResNet1d"),
            ("base_width", args.base_width),
            ("depth", args.depth),
        ]
        tu.print_run_config(args, world_size, device, extra_lines)

    # Load dataset, move to this rank's GPU
    dataset = tu.build_dataset(args.dataclass, project_root)
    dataset = tu.move_dataset_to_gpu(dataset, device)
    tu.print_main(rank, f"Dataset on {device}: "
                        f"{dataset.X.element_size() * dataset.X.nelement() / 1e9:.2f} GB")

    # Train/val split -- use a fixed seed for the generator so all ranks
    # agree on which indices belong to which split (otherwise different
    # ranks would train on different splits, defeating DDP).
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=gen,
    )

    # Build GPU-resident split views, then shard the training split
    # across ranks (each rank gets a non-overlapping slice).
    train_view = tu.make_split_view(dataset, train_ds.indices, device)
    val_view = tu.make_split_view(dataset, val_ds.indices, device)
    rank_shard = tu.partition_for_rank(train_view, rank, world_size, device)

    tu.print_main(rank, f"Rank {rank}/{world_size}: shard has {rank_shard.X.shape[0]:,} samples "
                        f"(global per-epoch: {rank_shard.X.shape[0] * world_size:,})")

    # Build model -- import deferred until after project_root is in sys.path
    from src.models_resnet1d import build_resnet1d
    model = build_resnet1d(base_width=args.base_width, depth=args.depth).to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    start_training = time.perf_counter()
    final_elapsed = None
    final_loss = None

    for epoch in range(args.epochs):
        if world_size > 1:
            dist.barrier()  # ensure all ranks start the epoch together for clean timing

        running_loss, elapsed = tu.train_one_epoch(
            model, optimizer, criterion, device, rank_shard,
            per_rank_batch_size, args.mixed_precision, amp_dtype, world_size,
        )

        if rank == 0:
            print(f"Epoch {epoch + 1} loss={running_loss:.3f} time={elapsed:.2f}s")
        final_elapsed = elapsed
        final_loss = running_loss

    elapsed_training = time.perf_counter() - start_training

    # Evaluation on rank 0 only -- DDP's model wrapper has the underlying
    # module accessible via .module; for single-GPU it's the model directly.
    if rank == 0:
        eval_model = model.module if world_size > 1 else model
        accuracy = tu.evaluate(eval_model, val_view, device,
                              batch_size=per_rank_batch_size)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Total training time: {elapsed_training:.2f}s")

        if args.result_csv:
            fieldnames = [
                "architecture", "world_size", "base_width", "depth",
                "batch_size", "per_rank_batch_size", "tf32", "precision",
                "epochs", "epoch_time_s", "total_time_s", "accuracy",
            ]
            precision = args.amp_dtype if args.mixed_precision else "fp32"
            row = {
                "architecture": "resnet1d",
                "world_size": world_size,
                "base_width": args.base_width,
                "depth": args.depth,
                "batch_size": args.batch_size,
                "per_rank_batch_size": per_rank_batch_size,
                "tf32": args.tf32,
                "precision": precision,
                "epochs": args.epochs,
                "epoch_time_s": f"{final_elapsed:.4f}",
                "total_time_s": f"{elapsed_training:.4f}",
                "accuracy": f"{accuracy:.4f}",
            }
            tu.write_result_csv(args.result_csv, fieldnames, row)
            print(f"Result row appended to {args.result_csv}")

    tu.cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
