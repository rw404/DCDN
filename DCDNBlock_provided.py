"""User-supplied production DCDN block, recorded on 2026-08-18.

The class body is preserved as provided. The user confirmed that production
uses one block initialized with ``x_i = x_0``, keeps ``x_0`` fixed, and passes
tensors shaped ``[B, 8, F]``;
the middle axis is the eight-dimensional piecewise-linear encoding of each
feature. The local FC path/projector is part of the production integration but
is intentionally excluded from the theorem-aligned public offline DCDN core.
Imports were added so the preserved class can be loaded for provenance and
isolated numerical tests.
"""

import torch
from torch import nn


class DCDNBlock(nn.Module):
    def __init__(
        self,
        features_count: int,
        div_eps: float = 1e-2,
        div_init_scale: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.block = nn.Linear(features_count, features_count)

        self.div_block = nn.Linear(features_count, features_count, bias=True)

        self.fc_block = nn.Sequential(
            nn.Linear(features_count, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.1),
        )

        self.projector = nn.Linear(features_count + 32, features_count)

        self.div_eps = div_eps

        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.block.weight)
            torch.nn.init.zeros_(self.block.bias)

            torch.nn.init.kaiming_normal_(self.div_block.weight)
            self.div_block.weight.data.mul_(div_init_scale)
            torch.nn.init.zeros_(self.div_block.bias)

            for layer in self.fc_block:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_normal_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

            torch.nn.init.kaiming_normal_(self.projector.weight)
            torch.nn.init.zeros_(self.projector.bias)

    def forward(self, x_0: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(x_0)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        safe_x_0 = torch.where(
            x_0.abs() > self.div_eps,
            x_0,
            sign * self.div_eps,
        )
        inv_x_0 = 1.0 / safe_x_0

        cross_x = x_i + x_0 * self.block(x_i)

        div_x = inv_x_0 * self.div_block(x_i)

        cross_div = cross_x + div_x

        fc_x = self.fc_block(x_i)

        x = torch.cat([cross_div, fc_x], dim=2)
        return self.projector(x)
