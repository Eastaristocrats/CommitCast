"""Lead/channel-specific online ridge sufficient statistics."""

from __future__ import annotations


class LeadChannelRidge:
    """Closed-form residual learner used by CommitCast.

    The object contains no ``torch.nn.Parameter`` and never constructs an
    optimizer. It updates only sufficient statistics from legally matured rows.
    """

    def __init__(
        self,
        *,
        leads: int,
        channels: int,
        features: int,
        ridge_lambda: float,
        beta_clip: float,
        device,
    ):
        import torch

        self.torch = torch
        self.leads = int(leads)
        self.features = int(features)
        self.ridge_lambda = float(ridge_lambda)
        self.beta_clip = float(beta_clip)
        self.matrix = torch.zeros(
            (leads, channels, features, features), dtype=torch.float32, device=device
        )
        self.vector = torch.zeros(
            (leads, channels, features), dtype=torch.float32, device=device
        )
        self.beta = torch.zeros(
            (leads, channels, features), dtype=torch.float32, device=device
        )
        self.eye = torch.eye(features, dtype=torch.float32, device=device)[None, :, :]
        self.last_matured = [0] * leads
        self.matured_seen = [0] * leads

    def update(self, lead: int, features, residual) -> None:
        count = max(1, int(features.shape[0]))
        self.matrix[lead] += self.torch.einsum(
            "bcf,bcg->cfg", features, features
        ) / count
        self.vector[lead] += self.torch.einsum(
            "bcf,bc->cf", features, residual
        ) / count
        self.matured_seen[lead] += count

    def solve(self, lead: int) -> None:
        solved = self.torch.linalg.solve(
            self.matrix[lead] + self.ridge_lambda * self.eye,
            self.vector[lead][..., None],
        ).squeeze(-1)
        self.beta[lead] = solved.clamp(min=-self.beta_clip, max=self.beta_clip)

    def active_mask(self, warmup: int, *, dtype, device):
        return self.torch.as_tensor(
            [count >= int(warmup) for count in self.matured_seen],
            dtype=dtype,
            device=device,
        )

    def predict(self, features):
        return self.torch.einsum("lcf,blcf->blc", self.beta, features)

