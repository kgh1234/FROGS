import os
import glob
import cv2
import torch
import numpy as np

# =========================
# Mask Utilities
# =========================
def _stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]

def _find_mask_path(mask_dir: str, image_name_or_path: str):
    stem = _stem(image_name_or_path)
    for ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
        cand = os.path.join(mask_dir, f"{stem}.{ext}")
        if os.path.isfile(cand):
            return cand
    # fallback for “mask” token
    for ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
        cands = glob.glob(os.path.join(mask_dir, f"{stem}*mask*.{ext}"))
        if cands:
            return cands[0]
    return None

def _load_binary_mask(mask_path: str, H: int, W: int, binary_threshold=32, invert=False, device="cuda"):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #m = cv2.flip(m, 0)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    
    if m.shape[0] == W and m.shape[1] == H:
        print(f"[MaskFix] Transposing mask (W,H)->(H,W): {mask_path}")
        m = m.T

    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if invert:
        m = 255 - m
    m = (m >= binary_threshold).astype(np.float32)
    return torch.from_numpy(m).to(device)  # (H, W)