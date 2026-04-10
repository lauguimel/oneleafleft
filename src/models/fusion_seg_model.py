"""Fusion segmentation model: Prithvi-EO encoder + auxiliary branch + UPerNet decoder.

Architecture:
  1. Prithvi-EO-2.0 (LoRA) encodes HLS chips → multi-scale feature maps
  2. Auxiliary MLP encodes extra bands (NDVI, NBR, static) → spatial feature map
  3. Feature maps concatenated at decoder input
  4. UPerNet decoder → 64×64 binary segmentation mask

This module wraps the TerraTorch model and adds the auxiliary fusion branch.
Used when tabular/auxiliary features need to be fused with the image encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AuxiliaryBranch(nn.Module):
    """Encode auxiliary spatial bands into a feature map for fusion.

    Input:  (B, C_aux, H, W) — e.g. 5 channels: NDVI, NBR, elev, slope, tc2000
    Output: (B, out_dim, H, W) — spatial feature map, same H×W
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionSegModel(nn.Module):
    """Wrapper that fuses TerraTorch Prithvi segmentation with auxiliary branch.

    The TerraTorch model handles: Prithvi encoder → necks → UPerNet decoder → logits.
    This wrapper:
      1. Runs the Prithvi encoder to get feature maps
      2. Runs the auxiliary branch
      3. Concatenates aux features to the last encoder feature map
      4. Runs the decoder on the fused features
      5. Returns logits (B, num_classes, H, W)
    """

    def __init__(
        self,
        terratorch_model: nn.Module,
        aux_in_channels: int = 5,
        aux_out_channels: int = 64,
    ):
        super().__init__()
        self.backbone = terratorch_model
        self.aux_branch = AuxiliaryBranch(aux_in_channels, aux_out_channels)
        # We'll concatenate aux features after the encoder

    def forward(
        self,
        image: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image: (B, C=6, T, H, W) HLS bands for Prithvi
            aux:   (B, C_aux, H, W) auxiliary bands (optional)

        Returns:
            logits: (B, num_classes, H, W)
        """
        # If no aux, just run the base model
        if aux is None:
            return self.backbone(image)

        # TODO: For now, run the full TerraTorch model as-is.
        # Fusion requires hooking into the encoder-decoder boundary,
        # which depends on TerraTorch internals. We'll implement this
        # once we confirm the base model works on Aqua.
        #
        # Fallback: concatenate aux to the input as extra bands
        # (requires adjusting backbone input channels — v2)
        return self.backbone(image)
