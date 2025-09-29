# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import glob
import cv2
import torch
import numpy as np
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# -----------------------
# Mask helpers
# -----------------------
def _stem(name_or_path: str) -> str:
    return os.path.splitext(os.path.basename(name_or_path))[0]

def _find_mask_path(mask_dir: str, image_name_or_path: str):
    """우선 동일 스템. 없으면 'mask' 토큰 포함 후보 탐색."""
    stem = _stem(image_name_or_path)
    strict = []
    for ext in ("png","PNG","jpg","JPG","jpeg","JPEG","webp","WEBP","bmp","BMP"):
        p = os.path.join(mask_dir, f"{stem}.{ext}")
        if os.path.isfile(p): strict.append(p)
    if strict:
        strict.sort(key=lambda p: (os.path.splitext(p)[1].lower() != ".png", len(p)))
        return strict[0]
    cands = []
    for ext in ("png","PNG","jpg","JPG","jpeg","JPEG","webp","WEBP","bmp","BMP"):
        cands += glob.glob(os.path.join(mask_dir, f"{stem}*.{ext}"))
    cands = [p for p in cands if "mask" in os.path.basename(p).lower()]
    if not cands: return None
    cands.sort(key=lambda p: len(os.path.basename(p)))
    return cands[0]

def _load_binary_mask(mask_path: str, H: int, W: int, thr: int = 128, invert: bool = False, device: str = "cuda"):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if invert:
        m = 255 - m
    m = (m.astype(np.float32) >= float(thr)).astype(np.float32)  # 0/1
    return torch.from_numpy(m).to(device)  # (H, W)


def save_tensor_png_rgb(t3chw: torch.Tensor, path: str):
    """t: (3,H,W), 0..1"""
    torchvision.utils.save_image(t3chw.clamp(0,1), path)

def save_tensor_png_rgba(t3chw: torch.Tensor, mask_hw: torch.Tensor, path: str):
    """t: (3,H,W) 0..1, mask: (H,W) {0,1} -> save RGBA PNG"""
    t = (t3chw.clamp(0,1) * 255.0).byte().detach().cpu()
    m = (mask_hw.clamp(0,1) * 255.0).byte().detach().cpu()
    # (H,W,4)
    rgba = torch.zeros((t.shape[1], t.shape[2], 4), dtype=torch.uint8)
    rgba[...,0] = t[0]
    rgba[...,1] = t[1]
    rgba[...,2] = t[2]
    rgba[...,3] = m
    # PIL 없이 torchvision로 RGBA 저장이 불가하므로 cv2 사용
    # cv2는 BGRA를 쓰므로 채널 순서 변환
    bgra = rgba.numpy()[..., [2,1,0,3]]
    cv2.imwrite(path, bgra)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh,
               mask_dir=None, mask_thr=128, mask_invert=False, save_rgba=False):
    render_root = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(render_root, "renders")
    gts_path = os.path.join(render_root, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    use_mask = mask_dir is not None and len(str(mask_dir)) > 0
    mask_cache = {}  # key: image_name(or stem) -> (H,W) tensor

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pkg = render(view, gaussians, pipeline, background,
                     use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = pkg["render"]       # (3,H,W) 0..1
        gt = view.original_image[0:3, :, :]  # (3,H,W) 0..1

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # ---------- Apply mask (optional) ----------
        if use_mask:
            H, W = rendering.shape[-2:]
            key_name = getattr(view, "image_name", None)
            if key_name is None or not isinstance(key_name, str) or len(key_name) == 0:
                key_name = _stem(getattr(view, "image_path", "frame"))

            if key_name not in mask_cache:
                mp = _find_mask_path(mask_dir, key_name)
                if mp is None:
                    # 마지막 시도: 스템 재시도
                    mp = _find_mask_path(mask_dir, _stem(key_name))
                if mp is None:
                    # 마스크 없으면 전부 1로 저장(경고)
                    print(f"[WARN] mask not found for {key_name}. Using all-ones.")
                    mask_cache[key_name] = torch.ones((H,W), device=rendering.device, dtype=torch.float32)
                else:
                    mask_cache[key_name] = _load_binary_mask(mp, H, W, thr=mask_thr, invert=mask_invert, device=rendering.device)

            mask2d = mask_cache[key_name]              # (H,W)
            mask3 = mask2d.unsqueeze(0)                # (1,H,W)
            rendering = rendering * mask3              # 배경 0
            gt = gt * mask3

        # ---------- Save ----------
        out_name = f"{idx:05d}.png"
        save_tensor_png_rgb(rendering, os.path.join(render_path, out_name))
        save_tensor_png_rgb(gt, os.path.join(gts_path, out_name))

        if use_mask and save_rgba:
            # 추가로 RGBA 버전도 저장
            save_tensor_png_rgba(rendering, mask2d, os.path.join(render_path, f"{idx:05d}_rgba.png"))
            save_tensor_png_rgba(gt, mask2d, os.path.join(gts_path, f"{idx:05d}_rgba.png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool,
                mask_dir=None, mask_thr=128, mask_invert=False, save_rgba=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,
                       dataset.train_test_exp, separate_sh,
                       mask_dir=mask_dir, mask_thr=mask_thr, mask_invert=mask_invert, save_rgba=save_rgba)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,
                       dataset.train_test_exp, separate_sh,
                       mask_dir=mask_dir, mask_thr=mask_thr, mask_invert=mask_invert, save_rgba=save_rgba)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # --- New args for masked rendering ---
    parser.add_argument("--mask_dir", type=str, default="", help="Mask directory; if set, outputs are masked (object-only).")
    parser.add_argument("--mask_binary_threshold", type=int, default=128, help="Mask binarization threshold (0-255).")
    parser.add_argument("--mask_invert", action="store_true", help="Invert mask if needed.")
    parser.add_argument("--save_rgba", action="store_true", help="Also save RGBA PNG (alpha=mask).")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,
                mask_dir=(args.mask_dir if len(args.mask_dir) > 0 else None),
                mask_thr=int(args.mask_binary_threshold),
                mask_invert=bool(args.mask_invert),
                save_rgba=bool(args.save_rgba))
