

import torch, os
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_mask_projection_with_centers(xy_proj, mask_img, save_path="debug/mask_check.png", point_size=5):
    """
    Visualize Gaussian 2D projections over mask image.

    Args:
        xy_proj (torch.Tensor): (N,2) projected coordinates (u,v)
        mask_img (torch.Tensor or np.ndarray): [H,W] or [1,H,W]
        save_path (str): output file path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert mask → numpy grayscale
    if isinstance(mask_img, torch.Tensor):
        mask_np = mask_img.detach().cpu().numpy()
    else:
        mask_np = mask_img

    if mask_np.ndim == 3:
        mask_np = mask_np[0]  # [1,H,W]

    H, W = mask_np.shape
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_np, cmap='gray', origin='upper')

    # Valid points only (inside image)
    u = torch.round(xy_proj[:, 0]).long()
    v = torch.round(xy_proj[:, 1]).long()

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_valid = u[valid].cpu().numpy()
    v_valid = v[valid].cpu().numpy()

    plt.scatter(u_valid, v_valid, s=point_size, c='red', alpha=0.7)

    plt.title(f"Mask projection check ({len(u_valid)} / {len(u)} visible)")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"[Saved] Projection visualization → {save_path}")
    
    