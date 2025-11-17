#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import glob
import torch
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor


# ======== PATHS ========
MIP_ROOT = '../../output_mipsplatting_ori/figurines/test'
CONFIG_FILE = "../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "../../GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAM_CKPT = "../../efficientvit_sam/weight/efficientvit_sam_xl1.pt"
PROMPT_JSON = f"{MIP_ROOT}/prompt_all.json"

CPU_ONLY = False
TINY_AREA_RATIO = 0.0002


# ========= Utils =========
def _normalize_caption(caption: str) -> str:
    caption = caption.strip()
    if not caption.endswith("."):
        caption += "."
    return caption.lower()


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def load_dino(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if (torch.cuda.is_available() and not cpu_only) else "cpu"
    model = build_model(args)
    ckpt = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    print("✅ GroundingDINO 로드 완료")
    return model


def xywhn_to_xyxy(box, W, H):
    box = box * torch.tensor([W, H, W, H])
    cx, cy, w, h = box
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    return np.array([int(x0), int(y0), int(x1), int(y1)])


@torch.no_grad()
def run_dino_once(model, image_tensor, caption, box_th=0.4, text_th=0.25, cpu_only=False):
    device = "cuda" if (torch.cuda.is_available() and not cpu_only) else "cpu"
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    caption = _normalize_caption(caption)

    outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    scores = logits.max(dim=1)[0]
    keep = scores > box_th
    if keep.sum().item() == 0:
        return []
    boxes = boxes[keep].cpu()
    scores = scores[keep].cpu()
    order = torch.argsort(scores, descending=True)
    return boxes[order]


def pick_best_mask(masks, H, W):
    if masks.ndim == 3:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        ys, xs = np.indices((H, W))
        scores = []
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.float32)
            denom = m.sum() + 1e-6
            cy = (ys * m).sum() / denom
            cx = (xs * m).sum() / denom
            dist = np.sqrt((cy - H / 2) ** 2 + (cx - W / 2) ** 2)
            scores.append((-dist, areas[i]))
        idx = int(np.lexsort((np.array(scores)[:, 1], np.array(scores)[:, 0]))[-1])
        return masks[idx]
    return masks


# ========= Main processing =========
def process_render_folder(render_dir, predictor, dino, prompts):
    """각 hemisphere_render 폴더 처리"""
    print(f"\n🎨 Processing {render_dir}")
    mask_dir = os.path.join(os.path.dirname(render_dir), "mask")
    os.makedirs(mask_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(render_dir, "*.png")) + glob.glob(os.path.join(render_dir, "*.jpg")))
    print(f"   └ {len(img_files)} images found")

    for img_path in tqdm(img_files, desc=f"[{os.path.basename(os.path.dirname(render_dir))}]"):
        fname = os.path.basename(img_path)
        out_mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")

        pil_img, img_tensor = load_image(img_path)
        img_np = np.array(pil_img)
        H, W = img_np.shape[:2]

        predictor.set_image(img_np)
        found = False

        for prompt in prompts:
            boxes = run_dino_once(dino, img_tensor, prompt)
            if len(boxes) == 0:
                continue

            for box in boxes:
                b = xywhn_to_xyxy(box, W, H)
                masks, _, _ = predictor.predict(box=b, multimask_output=True)
                mask = pick_best_mask(masks, H, W)
                mask_u8 = (mask.astype(np.uint8)) * 255

                if mask_u8.sum() < TINY_AREA_RATIO * (H * W) * 255:
                    continue

                cv2.imwrite(out_mask_path, mask_u8)
                found = True
                break
            if found:
                break

        if not found:
            black = np.zeros((H, W), dtype=np.uint8)
            cv2.imwrite(out_mask_path, black)

    print(f"✅ Saved masks to {mask_dir}")


def main():
    device = "cuda" if (torch.cuda.is_available() and not CPU_ONLY) else "cpu"
    print(f"🧠 Using device: {device}")

    # 모델 로드
    dino = load_dino(CONFIG_FILE, CHECKPOINT_PATH, cpu_only=CPU_ONLY)
    sam = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=False)
    sam.load_state_dict(torch.load(SAM_CKPT, map_location=device))
    sam = sam.to(device).eval()
    predictor = EfficientViTSamPredictor(sam)
    print("✅ SAM 모델 로드 완료\n")

    # prompt 로드
    with open(PROMPT_JSON, "r") as f:
        prompt_dict = json.load(f)

    render_dirs = sorted(glob.glob(os.path.join(MIP_ROOT, "*", "ours_30000", "hemisphere_render")))
    print(f"총 {len(render_dirs)}개의 scene 탐색됨")

    for render_dir in render_dirs:
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(render_dir)))
        prompts = prompt_dict.get(scene_name, [])
        if not prompts:
            print(f"⚠️ No prompt found for {scene_name}, skip.")
            continue
        process_render_folder(render_dir, predictor, dino, prompts)

    print("\n🎉 모든 scene 마스크 생성 완료!")


if __name__ == "__main__":
    main()
