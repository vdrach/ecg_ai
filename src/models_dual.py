"""
models_dual.py

Dual-branch architecture for ECG classification: a 1D CNN encoder
captures local morphology (P/QRS/T wave shapes), a Transformer encoder
captures longer-range temporal structure, and a learned fusion layer
combines their representations before the classifier head.

Both branches are parameterized independently so you can sweep size
without one branch dominating. The transformer is chosen specifically
(over LSTM/GRU) because it parallelizes along the time axis -- avoiding
the launch-overhead bottleneck that sequential RNNs would reintroduce.

Architecture:
  - CNN branch:
      Conv1d stem (2 -> cnn_width, stride 2) -> BN -> ReLU -> MaxPool
      Repeated cnn_depth times: Conv1d(cnn_width, cnn_width, k=3) + BN + ReLU
      Global average pool -> (batch, cnn_width)
  - Transformer branch:
      Conv1d projection (2 -> tf_width, stride 4) to reduce sequence length
      Add learned positional embedding
      tf_depth layers of TransformerEncoderLayer (tf_width dim, tf_heads heads)
      Global mean pool over time -> (batch, tf_width)
  - Fusion:
      Concatenate CNN + Transformer outputs -> Linear -> ReLU -> Linear -> classes
      The first Linear is the "learned fusion" layer
"""

import torch
import torch.nn as nn


class _CNNBranch(nn.Module):
    def __init__(self, in_channels=2, width=64, depth=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(
                nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            ))
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = width

    # in _CNNBranch.forward:
    def forward(self, x):
        x = x.transpose(1, 2)        # (batch, time, channels) -> (batch, channels, time)
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)



class _TransformerBranch(nn.Module):
    """Transformer encoder over the time axis.

    The input projection (Conv1d stride=4) downsamples the sequence to
    keep attention's quadratic cost manageable. With input length 1280,
    sequence after stride-4 projection is 320 -- the attention matrix
    is 320x320 per head, fully parallelizable on the GPU.
    """

    def __init__(self, in_channels=2, width=128, depth=4, heads=4,
                 seq_len_after_proj=320):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, width, kernel_size=7,
                             stride=4, padding=3)
        # Learned positional embedding -- simple and sufficient for fixed
        # sequence length. Sinusoidal would also work; not worth the
        # complexity for a workshop where the focus is scaling, not arch.
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len_after_proj, width))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_dim = width


    # in _TransformerBranch.forward:
    def forward(self, x):
        x = x.transpose(1, 2)        # (batch, time, channels) -> (batch, channels, time)
        x = self.proj(x)
        x = x.transpose(1, 2)        # now back to (batch, time/4, width) for attention
        x = x + self.pos_embed
        x = self.encoder(x)
        return x.mean(dim=1)
   
class DualBranchECG(nn.Module):
    """CNN + Transformer dual-branch ECG classifier with learned fusion."""

    def __init__(self, in_channels=2, num_classes=2,
                 cnn_width=64, cnn_depth=4,
                 tf_width=128, tf_depth=4, tf_heads=4,
                 fusion_hidden=256):
        super().__init__()
        self.cnn = _CNNBranch(in_channels=in_channels, width=cnn_width,
                              depth=cnn_depth)
        self.transformer = _TransformerBranch(
            in_channels=in_channels, width=tf_width, depth=tf_depth,
            heads=tf_heads,
        )

        fused_in = self.cnn.out_dim + self.transformer.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        tf_feat = self.transformer(x)
        fused = torch.cat([cnn_feat, tf_feat], dim=1)
        return self.fusion(fused)


def build_dual(cnn_width=64, cnn_depth=4,
              tf_width=128, tf_depth=4, tf_heads=4):
    """Convenience constructor for the training script."""
    return DualBranchECG(
        cnn_width=cnn_width, cnn_depth=cnn_depth,
        tf_width=tf_width, tf_depth=tf_depth, tf_heads=tf_heads,
    )
