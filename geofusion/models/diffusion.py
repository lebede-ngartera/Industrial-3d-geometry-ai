"""Diffusion-based shape generation and completion.

Implements a denoising diffusion probabilistic model (DDPM) for
generating 3D point cloud shapes, conditioned on text or category.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class PointCloudDenoiser(nn.Module):
    """Denoising network for point cloud diffusion.

    Predicts the noise added to a point cloud at a given timestep,
    optionally conditioned on a class or text embedding.
    """

    def __init__(
        self,
        point_dim: int = 3,
        num_points: int = 2048,
        hidden_dim: int = 256,
        time_dim: int = 128,
        condition_dim: int | None = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if condition_dim is not None:
            self.cond_proj = nn.Linear(condition_dim, hidden_dim)
            in_dim_first = point_dim + hidden_dim  # inject condition via concat
        else:
            self.cond_proj = None
            in_dim_first = point_dim

        # Point-wise processing layers
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_dim_first, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv1": nn.Conv1d(hidden_dim, hidden_dim, 1),
                        "norm1": nn.GroupNorm(8, hidden_dim),
                        "conv2": nn.Conv1d(hidden_dim, hidden_dim, 1),
                        "norm2": nn.GroupNorm(8, hidden_dim),
                        "time_proj": nn.Linear(hidden_dim, hidden_dim),
                    }
                )
            )

        self.output_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, point_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise.

        Args:
            x: (B, N, 3) noisy point cloud
            t: (B,) integer timesteps
            condition: (B, condition_dim) optional conditioning embedding

        Returns:
            noise_pred: (B, N, 3) predicted noise
        """
        B, N, C = x.shape
        time_emb = self.time_mlp(t)  # (B, hidden_dim)

        h = x.permute(0, 2, 1)  # (B, 3, N)

        # Inject condition
        if condition is not None and self.cond_proj is not None:
            cond = self.cond_proj(condition)  # (B, hidden_dim)
            cond = cond.unsqueeze(-1).expand(-1, -1, N)  # (B, hidden_dim, N)
            h = torch.cat([h, cond], dim=1)  # (B, 3+hidden_dim, N)

        h = self.input_conv(h)  # (B, hidden_dim, N)

        # Residual blocks with time injection
        for layer in self.layers:
            residual = h
            h = layer["norm1"](layer["conv1"](h))
            h = F.gelu(h)
            # Add time embedding
            t_proj = layer["time_proj"](time_emb).unsqueeze(-1)  # (B, hidden_dim, 1)
            h = h + t_proj
            h = layer["norm2"](layer["conv2"](h))
            h = F.gelu(h)
            h = h + residual

        noise_pred = self.output_conv(h)  # (B, 3, N)
        return noise_pred.permute(0, 2, 1)  # (B, N, 3)


class ShapeDiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model for 3D shape generation.

    Generates point cloud shapes via iterative denoising, optionally
    conditioned on text descriptions or category embeddings.
    """

    def __init__(
        self,
        num_points: int = 2048,
        point_dim: int = 3,
        hidden_dim: int = 256,
        condition_dim: int | None = 256,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_points = num_points
        self.num_timesteps = num_timesteps

        self.denoiser = PointCloudDenoiser(
            point_dim=point_dim,
            num_points=num_points,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
        )

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to point cloud.

        Args:
            x_start: (B, N, 3) clean point cloud
            t: (B,) timesteps
            noise: (B, N, 3) optional pre-sampled noise

        Returns:
            x_noisy: (B, N, 3) noisy point cloud
            noise: (B, N, 3) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def forward(
        self,
        points: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass: compute denoising loss.

        Args:
            points: (B, N, 3) clean point clouds
            condition: (B, cond_dim) optional conditioning

        Returns:
            Dictionary with loss value
        """
        B = points.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=points.device)

        x_noisy, noise = self.q_sample(points, t)
        noise_pred = self.denoiser(x_noisy, t, condition)

        loss = F.mse_loss(noise_pred, noise)

        return {"loss": loss, "noise_pred": noise_pred}

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        condition: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate new point cloud shapes via reverse diffusion.

        Args:
            batch_size: Number of shapes to generate
            condition: (B, cond_dim) optional conditioning
            device: Target device

        Returns:
            samples: (B, N, 3) generated point clouds
        """
        x = torch.randn(batch_size, self.num_points, 3, device=device)

        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

            noise_pred = self.denoiser(x, t, condition)

            beta = self.betas[t_idx]

            # Predict x_0
            x_0_pred = (
                x - self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred
            ) / self.sqrt_alphas_cumprod[t_idx]

            # Compute x_{t-1}
            if t_idx > 0:
                noise = torch.randn_like(x)
                sigma = self.posterior_variance[t_idx].sqrt()
                x = (
                    self.sqrt_recip_alphas[t_idx]
                    * (x - beta / self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred)
                    + sigma * noise
                )
            else:
                x = x_0_pred

        return x
