"""
train_dual.py

DDP-aware training script for the CNN + Transformer dual-branch ECG
classifier. Single-node multi-GPU via torchrun; manual GPU-resident
batching; rank-aware data sharding for strong scaling.

Same scaling story as train_resnet1d.py; the only differences are the
model class instantiated and the architecture-specific CLI flags.

Usage (single GPU):
    python train_dual.py --cnn_width 64 --cnn_depth 4 --tf_width 128 --tf_depth 4

Usage (4 GPUs, one node):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        train_dual.py --cnn_width 64 --cnn_depth 4 --tf_width 128 --tf_depth 4
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
    parser.add_argument("--cnn_width", type=int, default=64,
                        help="CNN branch channel width.")
    parser.add_argument("--cnn_depth", type=int, default=4,
                        help="CNN branch number of blocks after the stem.")
    parser.add_argument("--tf_width", type=int, default=128,
                        help="Transformer branch d_model.")
    parser.add_argument("--tf_depth", type=int, default=4,
                        help="Transformer branch number of encoder layers.")
    parser.add_argument("--tf_heads", type=int, default=4,
                        help="Transformer branch number of attention heads. "
                             "Must evenly divide tf_width.")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device = tu.setup_distributed()

    if args.tf_width % args.tf_heads != 0:
        if rank == 0:
            print(f"ERROR: --tf_width {args.tf_width} not divisible by "
                  f"--tf_heads {args.tf_heads}", file=sys.stderr)
        tu.cleanup_distributed(world_size)
        sys.exit(1)

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

    project_root = tu.find_project_root()
    sys.path.insert(0, str(project_root))

    if rank == 0:
        print(f"project root found: {project_root}")
        extra_lines = [
            ("architecture", "DualBranchECG (CNN + Transformer)"),
            ("cnn_width", args.cnn_width),
            ("cnn_depth", args.cnn_depth),
            ("tf_width", args.tf_width),
            ("tf_depth", args.tf_depth),
            ("tf_heads", args.tf_heads),
        ]
        tu.print_run_config(args, world_size, device, extra_lines)

    dataset = tu.build_dataset(args.dataclass, project_root)
    dataset = tu.move_dataset_to_gpu(dataset, device)
    tu.print_main(rank, f"Dataset on {device}: "
                        f"{dataset.X.element_size() * dataset.X.nelement() / 1e9:.2f} GB")

    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=gen,
    )

    train_view = tu.make_split_view(dataset, train_ds.indices, device)
    val_view = tu.make_split_view(dataset, val_ds.indices, device)
    rank_shard = tu.partition_for_rank(train_view, rank, world_size, device)

    tu.print_main(rank, f"Rank {rank}/{world_size}: shard has {rank_shard.X.shape[0]:,} samples "
                        f"(global per-epoch: {rank_shard.X.shape[0] * world_size:,})")

    from src.models_dual import build_dual
    model = build_dual(
        cnn_width=args.cnn_width, cnn_depth=args.cnn_depth,
        tf_width=args.tf_width, tf_depth=args.tf_depth, tf_heads=args.tf_heads,
    ).to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_training = time.perf_counter()
    final_elapsed = None
    final_loss = None

    for epoch in range(args.epochs):
        if world_size > 1:
            dist.barrier()

        running_loss, elapsed = tu.train_one_epoch(
            model, optimizer, criterion, device, rank_shard,
            per_rank_batch_size, args.mixed_precision, amp_dtype, world_size,
        )

        if rank == 0:
            print(f"Epoch {epoch + 1} loss={running_loss:.3f} time={elapsed:.2f}s")
        final_elapsed = elapsed
        final_loss = running_loss

    elapsed_training = time.perf_counter() - start_training

    if rank == 0:
        eval_model = model.module if world_size > 1 else model
        accuracy = tu.evaluate(eval_model, val_view, device,
                              batch_size=per_rank_batch_size)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Total training time: {elapsed_training:.2f}s")

        if args.result_csv:
            fieldnames = [
                "architecture", "world_size",
                "cnn_width", "cnn_depth", "tf_width", "tf_depth", "tf_heads",
                "batch_size", "per_rank_batch_size", "tf32", "precision",
                "epochs", "epoch_time_s", "total_time_s", "accuracy",
            ]
            precision = args.amp_dtype if args.mixed_precision else "fp32"
            row = {
                "architecture": "dual",
                "world_size": world_size,
                "cnn_width": args.cnn_width,
                "cnn_depth": args.cnn_depth,
                "tf_width": args.tf_width,
                "tf_depth": args.tf_depth,
                "tf_heads": args.tf_heads,
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
