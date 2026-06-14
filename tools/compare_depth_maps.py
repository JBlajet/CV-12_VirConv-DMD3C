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

    # Try P2 first (common KITTI format), fall back to P_rect_02
    p_line = None
    for prefix in ("P2:", "P_rect_02:"):
        matches = [l for l in lines if l.startswith(prefix)]
        if matches:
            p_line = matches[0]
            break

    if p_line is None:
        raise ValueError(f"No projection matrix line found in {calib_path}")

    P_vals = np.array(p_line.strip().split()[1:], dtype=np.float32)
    # Handle both 12-value (3x4) and 16-value (4x4) formats
    if len(P_vals) == 16:
        P_mat = P_vals.reshape(4, 4)[:3, :3]
    else:
        P_mat = P_vals.reshape(3, 4)[:3, :3]

    return P_mat.copy()


def load_velo2cam_transform(calib_path: Path):
    """Load R_velo2cam and T_cam from calibration file.

    Handles two KITTI calib formats:
      1) Separate 'R:' + 'T:' lines (demo.py style files)
      2) Single 'Tr_velo_to_cam:' line with 12 values = combined [R | T] matrix
    
    Returns RT matrix (3x4) = [R | T] for transforming lidar points.
    """
    with open(calib_path, "r") as f:
        lines = f.readlines()

    R_vals = None
    T_vals = None

    # Try Tr_velo_to_cam format first (actual KITTI calibration files)
    for line in lines:
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "Tr_velo_to_cam":
            vals = np.array([float(x) for x in value.split()], dtype=np.float32)
            # 12 values = combined [R | T] matrix (3x4)
            RT_matrix = vals.reshape(3, 4)
            R_vals = RT_matrix[:, :3]   # Rotation part (3x3)
            T_vals = RT_matrix[:, 3:4]  # Translation part (3x1 column vector)
            break

    # Fallback to separate R/T lines if Tr_velo_to_cam not found
    if R_vals is None or T_vals is None:
        for line in lines:
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "R":
                R_vals = np.array([float(x) for x in value.split()], dtype=np.float32).reshape(3, 3)
            elif key == "T":
                T_vals = np.array([float(x) for x in value.split()], dtype=np.float32).reshape(3, 1)

    if R_vals is None or T_vals is None:
        raise ValueError(f"Could not find transform lines in {calib_path}. Expected 'Tr_velo_to_cam:' (12 values) or separate 'R:' + 'T:' lines.")

    RT = np.hstack([R_vals, T_vals])  # 3x4 matrix [R | T]
    return RT


def project_lidar_to_depth(lidar_points: np.ndarray, K: np.ndarray, RT: np.ndarray):
    """Project LiDAR points to depth map using camera intrinsics.

    First transforms lidar from velodyne frame to camera rectified coordinate frame
    using R_velo2cam and T_cam from calibration file, then projects with K.
    Returns a sparse depth map (HxW) with NaN where no point exists.
    """
    H, W = 352, 1216
    depth_map = np.full((H, W), np.nan, dtype=np.float32)

    # Transform lidar from velodyne frame to camera rectified frame
    # Append homogeneous coordinate (1) and apply RT transform
    N = len(lidar_points)
    if N == 0:
        return depth_map

    homo_points = np.hstack([lidar_points, np.ones((N, 1))])  # Nx4
    cam_points = (RT @ homo_points.T).T  # Nx3 in camera coordinates

    # DEBUG: Check first few cam_points
    if N > 0:
        print(f"  [DEBUG] First 5 cam_points:\n{cam_points[:5]}")
        print(f"  [DEBUG] Cam points mean Z: {np.mean(cam_points[:, 2]):.2f}, min Z: {np.min(cam_points[:, 2]):.2f}, max Z: {np.max(cam_points[:, 2]):.2f}")

    x_img = cam_points[:, 0] / (cam_points[:, 2] + 1e-8) * K[0, 0] + K[0, 2]
    y_img = cam_points[:, 1] / (cam_points[:, 2] + 1e-8) * K[1, 1] + K[1, 2]

    ix = np.floor(x_img).astype(int)
    iy = np.floor(y_img).astype(int)

    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (cam_points[:, 2] > 0.5)
    print(f"  [DEBUG] Projection valid count: {valid.sum()} / {N}")

    ix, iy = ix[valid], iy[valid]
    depths = cam_points[valid, 2]

    # Keep the closest point at each pixel using np.isnan for correct NaN check
    for i in range(len(ix)):
        d = depths[i]
        if np.isnan(depth_map[iy[i], ix[i]]) or d < depth_map[iy[i], ix[i]]:
            depth_map[iy[i], ix[i]] = d

    return depth_map


def project_lidar_to_depth_simple(lidar_points: np.ndarray, K: np.ndarray):
    """Simplified projection without velodyne→camera transform.

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


# Legacy function kept for backward compatibility — uses simple projection without RT transform
def generate_sparse_depth(lidar_points: np.ndarray, K: np.ndarray):
    """Project LiDAR points to image plane and create sparse depth map (352x1216)."""
    return project_lidar_to_depth_simple(lidar_points, K)


# ---------------------------------------------------------------------------
# Data loading helpers (shared by both models)
# ---------------------------------------------------------------------------

def load_image(image_path: Path):
    """Load RGB image and crop to [352, 1216].

    Computes dynamic crop offsets from actual image size rather than hardcoding.
    Matches demo.py pattern: tp = img.shape[0] - 352, lp = (img.shape[1] - 1216) // 2
    """
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    
    # Dynamic crop offsets computed from actual image dimensions
    H_orig, W_orig = img.shape[:2]
    target_H, target_W = 352, 1216
    
    tp = (H_orig - target_H) // 2   # top padding
    lp = (W_orig - target_W) // 2   # left padding
    
    img = img[tp:tp + target_H, lp:lp + target_W]
    
    return img


def load_velodyne(velo_path: Path):
    """Load .bin velodyne file and return Nx4 array."""
    data = np.fromfile(str(velo_path), dtype=np.float32)
    return data.reshape(-1, 4)[:, :3]  # discard reflectance


# ---------------------------------------------------------------------------
# Depth visualization (shared, consistent for both models)
# ---------------------------------------------------------------------------

DEPTH_MIN = 0.5   # minimum valid distance in meters (for sparse LiDAR only)
DEPTH_MAX = 70.0  # maximum clipping range — matches DMD3C default

def visualize_depth(depth: np.ndarray, grayscale: bool = False, min_valid: float = DEPTH_MIN) -> np.ndarray:
    """Visualize depth map using OpenCV COLORMAP_JET or Grayscale with FIXED scale [min_valid, DEPTH_MAX].

    If grayscale=False (default):
        Blue  = close objects (~0.5m), Red/yellow = far objects (70m+).
    If grayscale=True:
        White = close objects (~0.5m), Black = far objects (70m+) or invalid.

    Invalid/NaN regions stay BLACK for clear contrast.
    """
    H, W = depth.shape
    mask = (depth >= min_valid) & (depth <= DEPTH_MAX) & ~np.isnan(depth)
    
    if not mask.any():
        return np.zeros((H, W, 3), dtype=np.uint8)

    if grayscale:
        vis_gray = np.zeros((H, W), dtype=np.uint8)  # Invalid regions stay BLACK
        clipped_depth = np.clip(depth[mask], min_valid, DEPTH_MAX).astype(np.float32)
        
        # Closer is whiter (255), farther is blacker (0)
        normed_values = (1.0 - (clipped_depth - min_valid) / (DEPTH_MAX - min_valid)) * 255.0
        vis_gray[mask] = normed_values.astype(np.uint8)
        return cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)
    else:
        # COLORMAP_JET mode
        vis_colored = np.zeros((H, W, 3), dtype=np.uint8)  # Invalid regions stay BLACK
        clipped_depth = np.clip(depth[mask], min_valid, DEPTH_MAX).astype(np.float32)
        
        # Fixed-scale normalization: map [min_valid, DEPTH_MAX] linearly to [0, 255]
        normed_values = ((clipped_depth - min_valid) / (DEPTH_MAX - min_valid)) * 255.0
        
        mask_flat = mask.flatten()
        vis_colored_flat = np.zeros((H * W, 3), dtype=np.uint8)
        
        # Reshape for applyColorMap which expects (1, N, 3) or (N, 1) uint8 array
        normed_uint8 = np.clip(normed_values, 0, 255).astype(np.float32).reshape(-1, 1)
        colored_flat = cv2.applyColorMap(normed_uint8.astype(np.uint8), cv2.COLORMAP_JET)
        
        vis_colored_flat[mask_flat] = colored_flat
        return vis_colored_flat.reshape(H, W, 3)


# Legacy function kept for backward compatibility — uses simple projection without RT transform
def project_lidar_to_depth_simple(lidar_points: np.ndarray, K: np.ndarray):
    pass


# ---------------------------------------------------------------------------
# DMD3C model loader & inference (adapted from demo.py)
# ---------------------------------------------------------------------------

def run_dmd3c(image: np.ndarray, sparse_depth: np.ndarray, K: np.ndarray):
    """Run DMD3C model and return predicted depth map."""
    # Import paths for DMD3C submodule (needed so BpOps.py stub is findable)
    dmd3c_dir = Path(__file__).parent / "DMD3C"
    sys.path.insert(0, str(dmd3c_dir))

    from models.BPNet import Net  # main model class
    
    device = torch.device("cpu")  # force CPU — uses local BpOps.py stub instead of CUDA extension
    
    # Load model weights
    checkpoint = torch.load(str(DMD3C_MODEL_PATH), map_location=device, weights_only=False)
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

    # DISP parameter is required by BPNet.forward(I, DISP, S, K) but can be None
    with torch.no_grad():
        output_list = model(img_tensor, None, sparse_tensor, K_tensor)  # returns list of tensors
    
    # The last element (pred0) is the full-resolution prediction
    pred_depth = output_list[-1] if isinstance(output_list, list) else output_list
    
    depth_np = pred_depth.squeeze().cpu().numpy()
    
    return depth_np.astype(np.float32)


# ---------------------------------------------------------------------------
# PENet model loader & inference (adapted from main.py + vis_utils.py)
# ---------------------------------------------------------------------------


def run_penet(image: np.ndarray, sparse_depth: np.ndarray, K: np.ndarray):
    """Run PENet model and return predicted depth map."""
    penet_dir = Path(__file__).parent / "PENet"
    sys.path.insert(0, str(penet_dir))

    from model import ENet  # main model class
    
    device = torch.device("cpu")  # force CPU — uses .cpu() fallbacks in model.py
    
    # Create args namespace matching PENet's argparse expectations
    class Args:
        network_model = 'pe'
        convolutional_layer_encoding = 'xyz'
        dilation_rate = 2
        freeze_backbone = False
    
    args = Args()

    checkpoint = torch.load(str(PENET_MODEL_PATH), map_location=device, weights_only=False)
    
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
    else:
        pred_depth = pred
    
    depth_np = pred_depth.squeeze().cpu().numpy()
    
    return depth_np.astype(np.float32)


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
    
    image = load_image(image_path)  # HxWxC, float32
    
    lidar_points = load_velodyne(velo_path)
    print(f"  [DEBUG] LiDAR points min/max: {lidar_points.min():.2f}/{lidar_points.max():.2f}")
    
    K = load_calibration(calib_path)
    RT = load_velo2cam_transform(calib_path)
    print(f"  [DEBUG] K: {K.flatten()}, RT: {RT.flatten()}")
    
    sparse_depth = project_lidar_to_depth(lidar_points, K, RT)

    print(f"  LiDAR points: {len(lidar_points)}, Valid depth pixels: {(sparse_depth > DEPTH_MIN).sum()}")
    if (sparse_depth > DEPTH_MIN).sum() == 0:
        print("  [WARNING] No valid sparse depth pixels found! Check projection and calibration.")
        # Try simple projection as fallback for debugging
        print("  [DEBUG] Trying simple projection...")
        sparse_depth = project_lidar_to_depth_simple(lidar_points, K)
        valid_count = (sparse_depth > DEPTH_MIN).sum() if sparse_depth is not None else 0
        print(f"  [DEBUG] Simple projection valid pixels: {valid_count}")

    # Run DMD3C
    print("  Running DMD3C...")
    dmd3c_depth = run_dmd3c(image, sparse_depth, K)
    
    # Debug: check what models returned
    if isinstance(dmd3c_depth, np.ndarray):
        valid_count = ((dmd3c_depth >= DEPTH_MIN) & (dmd3c_depth <= DEPTH_MAX)).sum()
        print(f"  DMD3C output shape={dmd3c_depth.shape}, min={dmd3c_depth.min():.2f}, max={dmd3c_depth.max():.2f}, valid pixels={valid_count}")

    # Run PENet  
    print("  Running PENet...")
    penet_depth = run_penet(image, sparse_depth, K)
    
    if isinstance(penet_depth, np.ndarray):
        valid_count_p = ((penet_depth >= DEPTH_MIN) & (penet_depth <= DEPTH_MAX)).sum()
        print(f"  PENET output shape={penet_depth.shape}, min={penet_depth.min():.2f}, max={penet_depth.max():.2f}, valid pixels={valid_count_p}")

    # Visualize all depth maps consistently (OpenCV COLORMAP_JET or Grayscale)
    # All use FIXED scale [DEPTH_MIN, DEPTH_MAX] for consistent color mapping:
    #   Grayscale mode: White = close objects (~0.5m), Black = far objects (70m+)
    vis_sparse = visualize_depth(sparse_depth, grayscale=True)
    vis_dmd3c = visualize_depth(dmd3c_depth, grayscale=True)
    vis_penet = visualize_depth(penet_depth, grayscale=True)

    # Original RGB image (resize to match depth map dimensions if needed)
    rgb_vis = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)  # BGR for OpenCV display
    
    H, W = 352, 1216
    
    # Resize RGB to match depth visualization size (if cropped differently)
    if rgb_vis.shape[:2] != (H, W):
        rgb_vis = cv2.resize(rgb_vis, (W, H))

    # Create side-by-side comparison: Original | DMD3C Predicted | PENet Predicted
    panel_labels = ["Original RGB", "DMD3C Pred.", "PENet Pred."]
    
    panels = [rgb_vis, vis_dmd3c, vis_penet]
    
    # Add text labels to each panel top-left corner
    labeled_panels = []
    for i, (panel, label) in enumerate(zip(panels, panel_labels)):
        labeled = panel.copy()
        cv2.putText(labeled, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        labeled_panels.append(labeled)

    # Concatenate horizontally with separators
    separator = np.zeros((H, 10, 3), dtype=np.uint8) + 64  # gray vertical bars
    
    comparison = np.hstack([panels[0], separator, panels[1], separator, panels[2]])

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
