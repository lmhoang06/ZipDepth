#!/usr/bin/env python3
"""
Test script to verify MosaicML Streaming integration in ZipDepth.
Uses the test shards in experiments/shard_experiment/shards_mds/ade20k and hrwsi.
"""

import os
import sys
import torch

from zipdepth.data import create_mds_dataloader, build_streams_from_sources
from zipdepth.model import create_model
from zipdepth.loss import ZipDepthLoss

def test_mds_integration():
    print("=" * 70)
    print(" TESTING MOSAICML STREAMING INTEGRATION IN ZIPDEPTH")
    print("=" * 70)
    
    ade_dir = "/mnt/d/MDE/experiments/shard_experiment/shards_mds/ade20k"
    hrwsi_dir = "/mnt/d/MDE/experiments/shard_experiment/shards_mds/hrwsi"
    
    if not (os.path.exists(ade_dir) and os.path.exists(hrwsi_dir)):
        print(f"Error: Test shards not found at {ade_dir} and {hrwsi_dir}")
        sys.exit(1)
        
    sources = f"{ade_dir},{hrwsi_dir}"
    print(f"Sources: {sources}")
    
    # 1. Test build_streams_from_sources
    streams = build_streams_from_sources(sources)
    print(f"Discovered {len(streams)} streams successfully.")
    for s in streams:
        print(f"  Stream local={s.local}, proportion={s.proportion}")
        
    # 2. Test create_mds_dataloader
    print("\nCreating MDS DataLoader with ZipDepth transforms (384x384, batch_size=4)...")
    loader, dataset = create_mds_dataloader(
        sources=sources,
        height=384,
        width=384,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
    )
    
    print(f"Dataset total samples: {len(dataset)}")
    print(f"DataLoader batches per epoch: {len(loader)}")
    
    # 3. Test retrieving first batch
    print("\nFetching first batch from DataLoader...")
    batch = next(iter(loader))
    
    images = batch["image"]
    depths = batch["depth"]
    domains = batch["domain"]
    paths = batch["path"]
    
    print("Batch successfully received:")
    print(f"  images shape: {images.shape}, dtype: {images.dtype}")
    print(f"  depths shape: {depths.shape}, dtype: {depths.dtype}")
    print(f"  domains: {domains}")
    print(f"  paths: {paths}")
    
    # Verify dtypes and shapes expected by ZipDepthTrainer
    assert images.ndim == 4 and images.shape[1] == 3 and images.shape[2:] == (384, 384)
    assert images.dtype == torch.uint8
    assert depths.ndim == 4 and depths.shape[1] == 1 and depths.shape[2:] == (384, 384)
    assert depths.dtype == torch.uint16
    
    # 4. Test forward pass and loss computation (ZipDepthTrainer compatibility)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTesting forward pass & loss on {device}...")
    
    model = create_model(variant="small", upsample_unfold=True).to(device)
    criterion = ZipDepthLoss(alpha_ssi=1.0, alpha_grad=2.0)
    
    # Matching ZipDepthTrainer lines 295-308
    norm_images = images.to(device).float() / 255.0
    teacher_depth = depths.to(device).float().div_(256.0)
    
    with torch.no_grad():
        student_depth = model(norm_images)
        loss, loss_dict = criterion(pred=student_depth, target=teacher_depth)
        
    print(f"Forward pass successful! Student output shape: {student_depth.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("\n" + "=" * 70)
    print(" ALL MOSAICML INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_mds_integration()
