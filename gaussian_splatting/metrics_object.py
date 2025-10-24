# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import cv2
import numpy as np
import re


def natural_key(s):
    """'img2.png' < 'img10.png' 처럼 숫자를 정수로 비교하는 정렬 키"""
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    render_files = sorted(os.listdir(renders_dir), key=natural_key)
    gt_files = sorted(os.listdir(gt_dir), key=natural_key)

    print(f"[DEBUG] first renders: {render_files[:3]}")
    print(f"[DEBUG] first gts    : {gt_files[:3]}")

    for fname_r, fname_g in zip(render_files, gt_files):
        render = Image.open(Path(renders_dir) / fname_r)
        gt = Image.open(Path(gt_dir) / fname_g)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname_r)
    return renders, gts, image_names


def evaluate(model_paths, mask_dir):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    mask_files = sorted(os.listdir(mask_dir), key=natural_key)
    selected_masks = [mask_files[i] for i in range(0, len(mask_files), 8)]
    print(f"Selected {len(selected_masks)} masks (every 8th frame)")
    print(f"[DEBUG] first masks: {mask_files[:3]}")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}
            method = sorted(os.listdir(test_dir))[-1]
            test_dir = Path(scene_dir) / "test"

            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

                # === mask 적용 ===
                if idx < len(selected_masks):
                    mask_path = os.path.join(mask_dir, selected_masks[idx])
                    if not os.path.exists(mask_path):
                        print(f"Mask not found: {mask_path}")
                        continue

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"Cannot read mask {mask_path}")
                        continue

                    h, w = renders[idx].shape[-2], renders[idx].shape[-1]
                    mask = cv2.resize(mask, (w, h)).astype(np.float32) / 255.0
                    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
                    mask_t = mask_t.expand_as(renders[idx])
                    render_masked = renders[idx] * mask_t
                    gt_masked = gts[idx] * mask_t
                else:
                    #print(f"No mask for frame {idx}")
                    render_masked = renders[idx]
                    gt_masked = gts[idx]
                PSNR = psnr(render_masked, gt_masked)
                if PSNR != float('inf'):
                    ssims.append(ssim(render_masked, gt_masked))
                    psnrs.append(PSNR)
                    lpipss.append(lpips(render_masked, gt_masked, net_type='vgg'))
                    #print('add')
                render_np = (render_masked.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                mask_np = mask_t.squeeze().detach().cpu().numpy()

                # (3, H, W) 형태일 경우 → (H, W, 3)로 변환
                if mask_np.ndim == 3 and mask_np.shape[0] == 3:
                    mask_np = np.transpose(mask_np, (1, 2, 0))

                # 흑백 마스크로 저장할 거라면 → 단일 채널로 변환
                if mask_np.ndim == 3:
                    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

                mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)
                
                #print(idx, render_np.shape, render_np.dtype, np.min(render_np), np.max(render_np))
                #print(idx, mask_np.shape, mask_np.dtype, np.min(mask_np), np.max(mask_np))

                save_dir = os.path.join(scene_dir, "masked_outputs")
                os.makedirs(save_dir, exist_ok=True)
                # 🔹 (3) 파일 이름 설정
                base_name = os.path.splitext(image_names[idx])[0]
                render_path = os.path.join(save_dir, f"{base_name}_render_masked.png")
                mask_path = os.path.join(save_dir, f"{base_name}_mask.png")

                # 🔹 (4) 저장
                cv2.imwrite(render_path, cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(mask_path, mask_np)
                # print(f"  SSIM : {ssim(render_masked, gt_masked)}")
                # print(f"  PSNR : {psnr(render_masked, gt_masked)}")
                # print(f"  LPIPS: {lpips(render_masked, gt_masked, net_type='vgg')}")
                # print("")

            print("SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
            print("PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
            print("LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
            print("")

            full_dict[scene_dir][method].update({
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item()
            })
            per_view_dict[scene_dir][method].update({
                "SSIM": {name: val for val, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "PSNR": {name: val for val, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "LPIPS": {name: val for val, name in zip(torch.tensor(lpipss).tolist(), image_names)}
            })

            with open(scene_dir + "/results_masked.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_masked.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, ":", e)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Metric evaluation (masked version, sorted)")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--mask_dir', '-mask', required=True, type=str, default="")
    args = parser.parse_args()
    evaluate(args.model_paths, args.mask_dir)
