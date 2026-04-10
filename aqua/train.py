"""Training script for Prithvi-EO segmentation on Aqua HPC.

Uses TerraTorch + Lightning for:
- Prithvi-EO-2.0 300M encoder with LoRA
- UPerNet decoder for 64×64 binary segmentation
- Focal loss (pixel-wise) for class imbalance
- Spatial-block-aware train/val/test splits (pre-split in HDF5)

Usage:
    python aqua/train.py --config aqua/config.yaml
    python aqua/train.py --config aqua/config.yaml --test-only
    python aqua/train.py --fast-dev-run  # smoke test on 2 batches

Expects HDF5 files in data/chips/:
    train_chips.h5, val_chips.h5, test_chips.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from models.chip_dataset import ChipDataset


def build_dataloaders(
    chips_dir: Path,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders from HDF5 chip files."""
    train_ds = ChipDataset(chips_dir / "train_chips.h5", augment=True)
    val_ds = ChipDataset(chips_dir / "val_chips.h5", augment=False)
    test_ds = ChipDataset(chips_dir / "test_chips.h5", augment=False)

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    print(f"Dataloaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_dl, val_dl, test_dl


def build_model():
    """Build Prithvi-EO segmentation model with LoRA via TerraTorch."""
    from terratorch.tasks import SemanticSegmentationTask
    from terratorch.datasets import HLSBands

    model_args = dict(
        backbone="prithvi_eo_v2_300",
        backbone_pretrained=True,
        backbone_num_frames=5,  # T=5 temporal composites
        backbone_bands=[
            HLSBands.BLUE,
            HLSBands.GREEN,
            HLSBands.RED,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
        necks=[
            {"name": "SelectIndices", "indices": [-1]},
            {"name": "ReshapeTokensToImage"},
        ],
        decoder="UperNetDecoder",
        decoder_channels=256,
        head_dropout=0.1,
        num_classes=2,  # background + deforestation
        # LoRA: parameter-efficient fine-tuning
        peft_config=dict(
            method="LORA",
            replace_qkv="qkv",
            peft_config_kwargs=dict(
                target_modules=["qkv.q_linear", "qkv.v_linear", "mlp.fc1", "mlp.fc2"],
                r=16,
                lora_alpha=16,
            ),
        ),
    )

    task = SemanticSegmentationTask(
        model_args=model_args,
        model_factory="EncoderDecoderFactory",
        loss="focal",
        loss_params={"gamma": 2.0, "alpha": 0.25},
        lr=6e-5,
        ignore_index=-1,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        lr_scheduler="cosine",
    )

    # Count parameters
    total = sum(p.numel() for p in task.parameters())
    trainable = sum(p.numel() for p in task.parameters() if p.requires_grad)
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable "
          f"({100*trainable/total:.1f}%)")

    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chips-dir", type=Path,
                        default=PROJECT_DIR / "data" / "chips")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("Prithvi-EO Deforestation Segmentation — Aqua Training")
    print("=" * 60)

    # Data
    train_dl, val_dl, test_dl = build_dataloaders(
        args.chips_dir, args.batch_size, args.num_workers,
    )

    # Model
    task = build_model()

    # Trainer
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=PROJECT_DIR / "models" / "checkpoints",
            filename="prithvi-seg-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir=str(PROJECT_DIR / "models" / "logs"),
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )

    if args.test_only:
        trainer.test(task, dataloaders=test_dl, ckpt_path=str(args.ckpt))
    else:
        trainer.fit(
            task,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=str(args.ckpt) if args.ckpt else None,
        )
        # Test after training
        trainer.test(task, dataloaders=test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
