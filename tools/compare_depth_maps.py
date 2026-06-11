#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare depth maps from DMD3C and PENet models on KITTI samples.

Usage:
    python tools/compare_depth_maps.py --frame_id 002666 [--save_dir output/comparison_vis]

This script runs both models on the same input (image + sparse LiDAR) and produces a side-by-side
comparison image with consistent visualization: Original Image | Sparse Depth | DMD3C Predicted | PENet Predicted.

Both depth maps are visualized using OpenCV COLORMAP_JET, clipped to [0, 70]m range for fair comparison.
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import os
import sys
import torch
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Paths (hardcoded)
# ---------------------------------------------------------------------------
BASE_ROOT = Path("output/visualize_data")
IMAGE_DIR = BASE_ROOT / "image_2"
VELODYNE_DIR = BASE_ROOT / "velodyne"
CALIB_DIR = BASE_ROOT / "calib"

DMD3C_MODEL_PATH = BASE_ROOT / "models" / "depth_completion_models" / "DMD3C_v2_weights.pth"
PENET_MODEL_PATH = BASE_ROOT / "models" / "depth_completion_models" / "PENet_weights.pth"


# ---------------------------------------------------------------------------
# Calibration loader (matches KITTI format)
# ---------------------------------------------------------------------------

def load_calibration(calib_path: Path):
    """Load calibration file and return camera intrinsics K (3x3)."""
    with open(calib_path, "r") as f:
        lines = f.readlines()

    P2_line = [l for l in lines if l.startswith("P2:")][0]
    P2_vals = np.array(P2_line.strip().split()[1:], dtype=np.float32)
    P2 = P2_vals.reshape(3, 4)
    K = P2[:3, :3].copy()

    # Adjust for center crop (image is cropped from 1242→1216 width and 390→352 height)
    K[0, 2] -= 13   # x-center shift
    K[1, 2] -= 11.5 # y-center shift

    return K


# ---------------------------------------------------------------------------
# Data loading helpers (shared by both models)
# ---------------------------------------------------------------------------

def load_image(image_path: Path):
    """Load RGB image and crop to [352, 1216]."""
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    # Center crop: height from 390→352 (crop ~19 top + ~19 bottom), width from 1242→1216 (crop 13 each side)
    img = img[19:371, 13:1229]
    return img


def load_velodyne(velo_path: Path):
    """Load .bin velodyne file and return Nx4 array."""
    data = np.fromfile(str(velo_path), dtype=np.float32)
    return data.reshape(-1, 4)[:, :3]  # discard reflectance


def generate_sparse_depth(lidar_points: np.ndarray, K: np.ndarray):
    """Project LiDAR points to image plane and create sparse depth map (352x1216)."""
    H, W = 352, 1216
    sparse_map = np.zeros((H, W), dtype=np.float32)

    # Transform lidar from velodyne frame to camera rectified frame
    # R_rect (from calibration file line "R0_rect") and Tr_velo_to_cam are needed
    return sparse_map  # simplified — full projection done below


def project_lidar_to_depth(lidar_points: np.ndarray, K: np.ndarray):
    """Project LiDAR points to depth map using camera intrinsics.

    Assumes lidar is already in the camera rectified coordinate frame (common for KITTI training data).
    Returns a 352x1216 sparse depth map with NaN where no point exists.
    """
    H, W = 352, 1216
    depth_map = np.full((H, W), np.nan, dtype=np.float32)

    x_img = lidar_points[:, 0] / (lidar_points[:, 2] + 1e-8) * K[0, 0] + K[0, 2]
    y_img = lidar_points[:, 1] / (lidar_points[:, 2] + 1e-8) * K[1, 1] + K[1, 2]

    ix = np.floor(x_img).astype(int)
    iy = np.floor(y_img).astype(int)

    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (lidar_points[:, 2] > 0.5)

    ix, iy = ix[valid], iy[valid]
    depths = lidar_points[valid, 2]

    # Keep the closest point at each pixel
    for i in range(len(ix)):
        d = depths[i]
        if depth_map[iy[i], ix[i]] == np.nan or d < depth_map[iy[i], ix[i]]:
            depth_map[iy[i], ix[i]] = d

    return depth_map


# ---------------------------------------------------------------------------
# Depth visualization (shared, consistent for both models)
# ---------------------------------------------------------------------------

DEPTH_MIN = 0.5   # minimum valid distance in meters
DEPTH_MAX = 70.0  # maximum clipping range — matches DMD3C default

def visualize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert depth map to color image using OpenCV COLORMAP_JET, clipped [DEPTH_MIN, DEPTH_MAX]."""
    vis = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    # Mask valid depths within range
    mask = (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)
    clipped_depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX)

    # Normalize to [0, 255] for colormap
    normed = np.zeros_like(clipped_depth)
    valid_count = mask.sum()
    if valid_count > 1:
        d_min_local = clipped_depth[mask].min()
        d_max_local = clipped_depth[mask].max()
        if d_max_local - d_min_local > 0:
            normed[mask] = (clipped_depth[mask] - d_min_local) / (d_max_local - d_min_local) * 255.0

    # Apply colormap only to valid regions
    vis_uint8 = np.uint8(normed)
    for h in range(vis.shape[0]):
        for w in range(vis.shape[1]):
            if mask[h, w]:
                vis[h, w] = cv2.applyColorMap(vis_uint8[h, w:h+1, w:w+1], cv2.COLORMAP_JET)[0, 0]

    # Faster vectorized version:
    normed_3ch = np.stack([normed] * 3, axis=-1)
    vis_colored = np.zeros((H_vis := depth.shape[0], W_vis := depth.shape[1], 3), dtype=np.uint8)
    
    for h in range(depth.shape[0]):
        row_mask = mask[h]
        if row_mask.any():
            vis_colored[h, row_mask] = cv2.applyColorMap(normed_3ch[h, row_mask].astype(np.uint8), cv2.COLORMAP_JET)

    return vis_colored


def visualize_depth_fast(depth: np.ndarray) -> np.ndarray:
    """Faster vectorized depth visualization using OpenCV COLORMAP_JET."""
    H, W = depth.shape
    
    # Create a float32 version for colormap input (values 0-255 expected by applyColorMap)
    normed = np.zeros((H, W), dtype=np.float32)
    
    mask = (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)
    clipped_depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX).copy()
    
    if mask.sum() > 1:
        d_min_local = clipped_depth[mask].min()
        d_max_local = clipped_depth[mask].max()
        if d_max_local - d_min_local > 0:
            normed[mask] = ((clipped_depth[mask] - d_min_local) / (d_max_local - d_min_local)) * 255.0
    
    # Convert to uint8 for applyColorMap
    normed_uint8 = np.uint8(np.clip(normed, 0, 255))
    
    # Apply colormap row by row (vectorized per-row)
    vis_colored = np.zeros((H, W, 3), dtype=np.uint8)
    for h in range(H):
        if mask[h].any():
            vis_colored[h] = cv2.applyColorMap(normed_uint8[h:h+1], cv2.COLORMAP_JET)[0]
    
    return vis_colored


# ---------------------------------------------------------------------------
# DMD3C model loader & inference (adapted from demo.py)
# ---------------------------------------------------------------------------

def run_dmd3c(image: np.ndarray, sparse_depth: np.ndarray, K: np.ndarray):
    """Run DMD3C model and return predicted depth map."""
    # Import paths for DMD3C submodule
    dmd3c_dir = Path(__file__).parent / "DMD3C"
    sys.path.insert(0, str(dmd3c_dir))

    from models.BPNet import Net  # main model class
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model weights
    checkpoint = torch.load(str(DMD3C_MODEL_PATH), map_location=device)
    state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    
    model = Net().to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Prepare inputs: image (HxWxC → CxHxW), sparse depth (1xHxW), K (3x3)
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    
    # Sparse depth: non-zero means valid LiDAR measurement
    sparse_d = np.where(sparse_depth > DEPTH_MIN, sparse_depth, 0).astype(np.float32)[np.newaxis, np.newaxis]
    sparse_tensor = torch.from_numpy(sparse_d).to(device)
    
    K_tensor = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_depth = model(img_tensor, sparse_tensor, K_tensor)  # returns depth map
    
    return pred_depth.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# PENet model loader & inference (adapted from main.py + vis_utils.py)
# ---------------------------------------------------------------------------

def run_penet(image: np.ndarray, sparse_depth: np.ndarray, K: np.ndarray):
    """Run PENet model and return predicted depth map."""
    penet_dir = Path(__file__).parent / "PENet"
    sys.path.insert(0, str(penet_dir))

    from model import ENet  # main model class
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create args namespace matching PENet's argparse expectations
    class Args:
        network_model = 'pe'
        convolutional_layer_encoding = 'xyz'
        dilation_rate = 2
        freeze_backbone = False
    
    args = Args()

    checkpoint = torch.load(str(PENET_MODEL_PATH), map_location=device)
    
    # Handle different checkpoint formats (some have 'state_dict', others direct weights)
    if isinstance(checkpoint, dict):
        for key in ['net', 'state_dict', 'model_state']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            # Try last item or first weight-like key
            state_dict = {k.replace('module.', ''): v for k, v in list(checkpoint.items()) 
                         if isinstance(v, torch.Tensor)}
    else:
        state_dict = checkpoint

    model = ENet(args).to(device)
    
    # Filter out module. prefix if present
    filtered_state = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        filtered_state[new_k] = v
    
    try:
        model.load_state_dict(filtered_state, strict=False)
    except RuntimeError as e:
        print(f"  [PENet] Warning loading weights (strict=False): {e}")
    
    model.eval()

    # Prepare inputs for PENet: expects dict with 'rgb', 'd' keys
    rgb_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    
    sparse_d = np.where(sparse_depth > DEPTH_MIN, sparse_depth, 0).astype(np.float32)[np.newaxis, np.newaxis]
    depth_tensor = torch.from_numpy(sparse_d).to(device)

    # Position maps (normalized u,v coordinates for geometric encoding)
    H, W = image.shape[:2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    uu_norm = (uu - K[0, 2]) / K[0, 0]
    vv_norm = (vv - K[1, 2]) / K[1, 1]
    
    pos_tensor_u = torch.from_numpy(uu_norm).unsqueeze(0).float().to(device)
    pos_tensor_v = torch.from_numpy(vv_norm).unsqueeze(0).float().to(device)

    # Build input dict matching PENet's forward() signature
    batch_input = {
        'rgb': rgb_tensor,
        'd': depth_tensor,
        'position': torch.stack([pos_tensor_u, pos_tensor_v], dim=1),  # Bx2xHxW
        'K': torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device),
    }

    with torch.no_grad():
        pred = model(batch_input)
    
    if isinstance(pred, (list, tuple)):
        pred_depth = pred[-1]  # last element is final depth prediction
    
    return pred_depth.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

def compare_single_frame(frame_id: str, save_dir: Path | None = None):
    """Run both models on a single KITTI frame and produce comparison visualization."""
    
    # Validate inputs exist
    image_path = IMAGE_DIR / f"{frame_id}.png"
    velo_path = VELODYNE_DIR / f"{frame_id}.bin"
    calib_path = CALIB_DIR / f"{frame_id}.txt"

    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}")
        return None
    if not velo_path.exists():
        print(f"[ERROR] Velodyne file not found: {velo_path}")
        return None
    if not calib_path.exists():
        print(f"[ERROR] Calibration file not found: {calib_path}")
        return None

    # Load data
    print(f"  Loading frame {frame_id}...")
    
    image = load_image(image_path)  # HxWxC, uint8
    
    lidar_points = load_velodyne(velo_path)
    
    K = load_calibration(calib_path)
    
    sparse_depth = project_lidar_to_depth(lidar_points, K)

    print(f"  LiDAR points: {len(lidar_points)}, Valid depth pixels: {(sparse_depth > DEPTH_MIN).sum()}")

    # Run DMD3C
    print("  Running DMD3C...")
    dmd3c_depth = run_dmd3c(image, sparse_depth, K)
    
    # Run PENet  
    print("  Running PENet...")
    penet_depth = run_penet(image, sparse_depth, K)

    # Visualize all depth maps consistently (OpenCV COLORMAP_JET, [0.5, 70]m clip)
    vis_sparse = visualize_depth_fast(sparse_depth)
    vis_dmd3c = visualize_depth_fast(dmd3c_depth)
    vis_penet = visualize_depth_fast(penet_depth)

    # Original RGB image (resize to match depth map dimensions if needed)
    rgb_vis = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)  # BGR for OpenCV display
    
    H, W = 352, 1216
    
    # Resize RGB to match depth visualization size (if cropped differently)
    if rgb_vis.shape[:2] != (H, W):
        rgb_vis = cv2.resize(rgb_vis, (W, H))

    # Create side-by-side comparison: Original | Sparse Depth | DMD3C Predicted | PENet Predicted
    panel_labels = ["Original RGB", "Sparse LiDAR", "DMD3C Pred.", "PENet Pred."]
    
    panels = [rgb_vis, vis_sparse, vis_dmd3c, vis_penet]
    
    # Add text labels to each panel top-left corner
    labeled_panels = []
    for i, (panel, label) in enumerate(zip(panels, panel_labels)):
        labeled = panel.copy()
        cv2.putText(labeled, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        labeled_panels.append(labeled)

    # Concatenate horizontally with separators
    separator = np.zeros((H, 10, 3), dtype=np.uint8) + 64  # gray vertical bars
    
    comparison = np.hstack([panels[0], separator] + 
                          [sep for p in panels[1:] for sep in [separator, p]])

    print(f"  Comparison image shape: {comparison.shape}")
    
    if save_dir is not None:
        os.makedirs(str(save_dir), exist_ok=True)
        out_path = save_dir / f"{frame_id}_depth_comparison.png"
        cv2.imwrite(str(out_path), comparison)
        print(f"  Saved to: {out_path}")
        
        # Also save individual depth maps for inspection
        ind_dir = save_dir / "individual_depths"
        os.makedirs(str(ind_dir), exist_ok=True)
        
        np.save(ind_dir / f"{frame_id}_sparse.npy", sparse_depth)
        np.save(ind_dir / f"{frame_id}_dmd3c_pred.npy", dmd3c_depth)
        np.save(ind_dir / f"{frame_id}_penet_pred.npy", penet_depth)
        
        # Save colored versions too
        cv2.imwrite(str(ind_dir / f"{frame_id}_sparse_colored.png"), vis_sparse)
        cv2.imwrite(str(ind_dir / f"{frame_id}_dmd3c_colored.png"), vis_dmd3c)
        cv2.imwrite(str(ind_dir / f"{frame_id}_penet_colored.png"), vis_penet)

    return comparison


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare DMD3C and PENet depth map predictions on KITTI samples."
    )
    parser.add_argument("--frame_id", type=str, required=True,
                       help="Frame ID (e.g., 002666)")
    parser.add_argument("--save_dir", type=Path, default=None,
                       help="Directory to save comparison images. Default: output/comparison_vis/<frame_id>")
    
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = BASE_ROOT.parent / "comparison_vis" / args.frame_id
    
    print(f"DMD3C model path: {DMD3C_MODEL_PATH}")
    print(f"PENet  model path: {PENET_MODEL_PATH}")
    
    result = compare_single_frame(args.frame_id, args.save_dir)
    
    if result is not None:
        print("\nDone! Comparison saved.")
    else:
        print("\nFailed to produce comparison.")


if __name__ == "__main__":
    main()
