#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import time
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator

# -----------------------------
# 사용자 경로/파라미터
# -----------------------------
# 이미지 폴더
IMG_DIR = "/workspace/gahyeon/dataset/garden/images"
# gdino_env에서 미리 생성해 둔 박스 JSON 폴더 (gdino_boxes.py 결과)
BOX_DIR = "/workspace/gahyeon/boxes"
# EfficientViT-SAM 가중치
CKPT_SAM = "/workspace/gahyeon/efficientvit_sam/weight/efficientvit_sam_xl1.pt"
# 출력 폴더
OUT_DIR = "/workspace/gahyeon/efficientvit_sam/outputs/table_mask_time_dino"

# 마스크 생성/후처리 하이퍼파라미터
MAX_SHORT_SIDE = 1024
POINTS_PER_SIDE = 16
CROP_N_LAYERS = 0

BIN_THR = 0.5     # 최종 이진화 임계값 (0~1)
MORPH_K = 5       # 모폴로지 커널 크기
IOU_THR = 0.40    # 박스-마스크 IoU 임계값
MIN_AREA = 128    # 너무 작은 마스크 제거

# phrases에서 골라낼 타겟 키워드 (gdino_boxes.py --phrase-filter 와 동일하게 맞추면 깔끔)
TARGET_KEY = "table"

# -----------------------------
# 유틸
# -----------------------------
def resize_keep_aspect(img: np.ndarray, max_short_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    short = min(h, w)
    if short <= max_short_side:
        return img, 1.0
    scale = max_short_side / float(short)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img_r, scale

def morph_cleanup(mask_u8: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8

def iou_mask_box(mask_u8: np.ndarray, box_xyxy: np.ndarray) -> float:
    """mask(0/255)와 box(x1,y1,x2,y2)의 IoU를 계산"""
    h, w = mask_u8.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1 = np.clip(x1, 0, w - 1); x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1); y2 = np.clip(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_mask = np.zeros_like(mask_u8, dtype=np.uint8)
    box_mask[y1:y2+1, x1:x2+1] = 255

    inter = np.logical_and(mask_u8 > 0, box_mask > 0).sum()
    union = np.logical_or(mask_u8 > 0, box_mask > 0).sum()
    return float(inter) / float(union + 1e-6)

def select_masks_by_boxes(
    masks_small: List[Dict],
    img_small_shape,
    img_full_shape,
    boxes_xyxy_full: np.ndarray
) -> np.ndarray | None:
    """AMG 생성 마스크 중, 주어진 박스들과 충분히 겹치는 마스크만 OR-합치기"""
    H, W = img_full_shape[:2]

    combined = np.zeros((H, W), dtype=np.uint8)
    any_selected = False

    for m in masks_small:
        m_small = m["segmentation"].astype(np.uint8)  # (h_s, w_s) 0/1
        m_full = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
        m_u8 = (m_full * 255).astype(np.uint8)

        if m_u8.sum() < MIN_AREA:
            continue

        ok = False
        for box in boxes_xyxy_full:
            if iou_mask_box(m_u8, box) >= IOU_THR:
                ok = True
                break

        if ok:
            combined = np.maximum(combined, m_u8)
            any_selected = True

    if not any_selected:
        return None
    return combined

def load_boxes_from_json(json_path: str, target_key: str) -> np.ndarray | None:
    """
    gdino_boxes.py가 저장한 JSON에서 target_key(예: 'table')에 해당하는 박스만 반환.
    스키마 호환: boxes_xyxy 또는 boxes 키 지원.
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    boxes = data.get("boxes_xyxy", data.get("boxes", []))
    phrases = data.get("phrases", [])

    if not boxes or not phrases:
        return None

    sel = [i for i, p in enumerate(phrases) if target_key.lower() in str(p).lower()]
    if not sel:
        return None

    boxes = np.array(boxes, dtype=float)
    return boxes[sel]

# -----------------------------
# 개별 이미지 처리
# -----------------------------
def process_one_image(img_path: str, out_path: str, sam_amg, device: str):
    start_time = time.time()

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Cannot read image: {img_path}")
        return None
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1) EfficientViT-SAM: 다수 마스크 생성 (축소본)
    img_small, _ = resize_keep_aspect(img_rgb, MAX_SHORT_SIDE)
    masks_small = sam_amg.generate(img_small)
    if len(masks_small) == 0:
        print(f"❌ No masks produced for {img_path}")
        return None

    # 2) JSON에서 'table' 박스 로드
    stem = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(BOX_DIR, f"{stem}.json")
    boxes_sel = load_boxes_from_json(json_path, TARGET_KEY)
    if boxes_sel is None or len(boxes_sel) == 0:
        print(f"❌ No '{TARGET_KEY}' boxes in JSON: {json_path}")
        return None

    # 3) 박스와 IoU로 마스크 선택 → 합치기
    mask_u8 = select_masks_by_boxes(
        masks_small=masks_small,
        img_small_shape=img_small.shape,
        img_full_shape=img_rgb.shape,
        boxes_xyxy_full=boxes_sel
    )
    if mask_u8 is None:
        print(f"❌ No masks matched '{TARGET_KEY}' boxes for {img_path}")
        return None

    # 4) 이진화 + 후처리
    mask_u8 = ((mask_u8 > int(BIN_THR * 255)) * 255).astype(np.uint8)
    mask_u8 = morph_cleanup(mask_u8, k=MORPH_K)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mask_u8)

    elapsed = time.time() - start_time
    print(f"✅ Saved {out_path} (time {elapsed:.2f}s, boxes {len(boxes_sel)}, masks {len(masks_small)})")
    return elapsed

# -----------------------------
# 메인
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # EfficientViT-SAM 로드
    sam = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=False)
    sam.load_state_dict(torch.load(CKPT_SAM, map_location=device))
    sam = sam.to(device).eval()

    sam_amg = EfficientViTSamAutomaticMaskGenerator(
        sam,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=32,
        crop_n_layers=CROP_N_LAYERS,
        crop_n_points_downscale_factor=2,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        box_nms_thresh=0.7,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    img_files.sort()

    total_time, count = 0.0, 0
    for fname in img_files:
        img_path = os.path.join(IMG_DIR, fname)
        out_path = os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}_{TARGET_KEY}_mask.png")
        print(f"🔹 Processing {img_path}")
        elapsed = process_one_image(img_path, out_path, sam_amg, device)
        if elapsed is not None:
            total_time += elapsed
            count += 1

    if count > 0:
        print(f"\n📊 Average time per image: {total_time / count:.2f}s over {count} images")

if __name__ == "__main__":
    main()
