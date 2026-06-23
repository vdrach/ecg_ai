"""
models_resnet1d.py

Parametric 1D ResNet for ECG classification. Designed to be properly
compute-bound at moderate-to-large sizes so DDP scaling experiments show
clean speedup curves (compute time dominates, gradient all_reduce is a
small fraction).

Architecture:
  - Input: (batch, 2, 1280) -- ECG signal, 2 channels, 1280 timesteps
  - Stem: single Conv1d(2 -> base_width, kernel=7, stride=2) + BN + ReLU + MaxPool
  - 4 stages of residual blocks at widths [base_width, 2*, 4*, 8*]
  - Number of blocks per stage controlled by `depth` parameter
  - Global average pooling over the time axis
  - Linear classifier head to num_classes (2 for ECG normal/abnormal)

Two parameters control size:
  - `base_width` (1D analogue of ResNet's channel count): roughly squares
    the FLOP cost when doubled. Sensible range: 32 (small) to 256 (large).
  - `depth` (number of residual blocks per stage): linearly scales FLOPs
    and parameter count. Sensible range: 2 (small) to 6 (large).
"""

import torch.nn as nn


class _BasicBlock1d(nn.Module):
    """Standard 1D ResNet basic block: two 3x3 (well, 3x1) convs +
    skip connection. Downsampling (stride=2) on the first conv when
    entering a new stage; identity skip otherwise.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 conv on the skip when channel count or stride changes,
        # otherwise identity. Standard ResNet pattern.
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet1d(nn.Module):
    """Parametric 1D ResNet."""

    def __init__(self, in_channels=2, num_classes=2,
                 base_width=64, depth=2):
        super().__init__()

        # Stem: aggressive downsampling like the original 2D ResNet
        # to bring sequence length down quickly before deeper layers.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 4 stages, widths doubling each time, first block of each
        # stage (except the first stage) does stride-2 downsampling.
        stage_widths = [base_width * (2 ** i) for i in range(4)]
        stages = []
        in_w = base_width
        for stage_idx, out_w in enumerate(stage_widths):
            for block_idx in range(depth):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                stages.append(_BasicBlock1d(in_w, out_w, stride=stride))
                in_w = out_w
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool over time
        self.classifier = nn.Linear(stage_widths[-1], num_classes)


    def forward(self, x):
        # Dataset stores tensors as (batch, time, channels) -- transpose to
        # (batch, channels, time) for Conv1d/BatchNorm1d which expect channels
        # in dim 1.
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

def build_resnet1d(base_width=64, depth=2):
    """Convenience constructor for the training script."""
    return ResNet1d(base_width=base_width, depth=depth)
