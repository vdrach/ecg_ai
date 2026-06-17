import time

from pathlib import Path
import sys

project_root = Path.cwd()

while not (project_root / "src").exists():
    project_root = project_root.parent

sys.path.insert(
    0,
    str(project_root)
)

print(f"project root found: {project_root}")

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import ECGDataset
from src.dataset import CachedECGDataset
from src.dataset import FastCachedECGDataset
from src.models import MLP
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cpu", "mps", "cuda"],
)
parser.add_argument(
    "--dataclass",
    default="fast",
    choices=["slow", "medium", "fast"],
)
args = parser.parse_args()

if args.device == "auto":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

else:
    device = torch.device(args.device)



if args.dataclass == 'fast':
    dataset = FastCachedECGDataset(
        "data/processed"
    )
elif args.dataclass == 'medium':
     dataset = CachedECGDataset(
        "data/processed"
    )
elif args.dataclass == 'slow':
    dataset = ECGDataset(
            "data/processed"
    )
n = len(dataset)

print(
    f"Training on {device} "
    f"with {len(dataset):,} samples "
    f"and dataclass: {args.dataclass}"
)

train_size = int(0.8 * n)
val_size = n - train_size

train_ds, val_ds = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
)


train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
)

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

epochs = 10
start_training = time.perf_counter()

for epoch in range(epochs):

    model.train()
    
    running_loss = 0
    start = time.perf_counter()

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(
            logits,
            y,
        )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"Epoch {epoch+1} "
        f"loss={running_loss:.3f} "
        f"time={elapsed:.2f}s"
    )


    model.eval()


elapsed_training = time.perf_counter() - start_training 


correct = 0
total = 0

with torch.no_grad():

    for x, y in val_loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(1)

        correct += (
            pred == y
        ).sum().item()

        total += len(y)

print(f"Accuracy: {correct/total:.3f}")
