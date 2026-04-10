"""PyTorch Dataset for HLS chips + Hansen segmentation masks.

Reads from HDF5 files produced by scripts/extract_chips_hls.py.
Returns:
  - image: (T, C, H, W) float32 tensor — T timesteps, C=6 HLS bands (Prithvi input)
  - mask:  (H, W) long tensor — binary segmentation target
  - aux:   (C_aux, H, W) float32 tensor — auxiliary bands (NDVI, NBR, static)
           broadcast-ready for fusion with Prithvi features
  - meta:  dict with pid, pred_year, lon, lat

HDF5 layout (from extract_chips_hls.py):
  images: (N, T=5, C=11, H=64, W=64) — 6 HLS + NDVI + NBR + elev + slope + tc2000
  masks:  (N, H=64, W=64) — binary
  pid, pred_year, lon, lat: (N,) metadata
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# HLS band indices in the HDF5 (B2, B3, B4, B5, B6, B7, NDVI, NBR, elev, slope, tc2000)
HLS_BAND_INDICES = list(range(6))       # B2..B7 → channels 0-5
AUX_BAND_INDICES = list(range(6, 11))   # NDVI, NBR, elevation, slope, treecover2000


class ChipDataset(Dataset):
    """HLS chip dataset for semantic segmentation."""

    def __init__(
        self,
        h5_path: str | Path,
        augment: bool = False,
    ):
        self.h5_path = Path(h5_path)
        self.augment = augment

        # Read metadata eagerly (small), keep images lazy
        with h5py.File(self.h5_path, "r") as f:
            self.n = f["images"].shape[0]
            self.pids = f["pid"][:]
            self.pred_years = f["pred_year"][:]
            self.lons = f["lon"][:]
            self.lats = f["lat"][:]

        # HDF5 handle opened lazily per worker (multiprocessing safe)
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> dict:
        self._open()

        # Read from HDF5: (T, C_all, H, W) and (H, W)
        image_all = self._h5["images"][idx].astype(np.float32)  # (T, 11, 64, 64)
        mask = self._h5["masks"][idx].astype(np.int64)           # (64, 64)

        # Split into HLS bands (for Prithvi) and auxiliary bands
        image_hls = image_all[:, HLS_BAND_INDICES, :, :]  # (T, 6, H, W)
        # Auxiliary: use last timestep only (static + indices)
        image_aux = image_all[-1, AUX_BAND_INDICES, :, :]  # (C_aux, H, W)

        # Handle NaN → 0
        image_hls = np.nan_to_num(image_hls, nan=0.0)
        image_aux = np.nan_to_num(image_aux, nan=0.0)

        # Augmentations (random flip + rotation)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image_hls = image_hls[:, :, :, ::-1].copy()
                image_aux = image_aux[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                image_hls = image_hls[:, :, ::-1, :].copy()
                image_aux = image_aux[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
            # Random 90° rotation
            k = np.random.randint(4)
            if k > 0:
                image_hls = np.rot90(image_hls, k, axes=(2, 3)).copy()
                image_aux = np.rot90(image_aux, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # Prithvi expects (C, T, H, W) — transpose from (T, C, H, W)
        image_hls = image_hls.transpose(1, 0, 2, 3)  # (6, T, H, W)

        return {
            "image": torch.from_numpy(image_hls),
            "mask": torch.from_numpy(mask),
            "aux": torch.from_numpy(image_aux),
            "meta": {
                "pid": int(self.pids[idx]),
                "pred_year": int(self.pred_years[idx]),
                "lon": float(self.lons[idx]),
                "lat": float(self.lats[idx]),
            },
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
