import torch
import torch.nn as nn


class Fusion(nn.Module):
    """Weighted BiFPN feature fusion used in CLBD-YOLO."""

    def __init__(self, inc_list, fusion='bifpn'):
        super().__init__()

        if fusion != 'bifpn':
            raise ValueError(
                f"CLBD-YOLO uses only BiFPN fusion, but got fusion='{fusion}'."
            )

        self.fusion = fusion

        self.fusion_weight = nn.Parameter(
            torch.ones(len(inc_list), dtype=torch.float32),
            requires_grad=True
        )

        self.relu = nn.ReLU()
        self.epsilon = 1e-4

    def forward(self, x):
        fusion_weight = self.relu(self.fusion_weight.clone())

        fusion_weight = fusion_weight / (
            torch.sum(fusion_weight, dim=0) + self.epsilon
        )

        out = torch.sum(
            torch.stack(
                [
                    fusion_weight[i] * x[i]
                    for i in range(len(x))
                ],
                dim=0
            ),
            dim=0
        )

        return out
