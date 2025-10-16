import os
import cv2
import torch
import clip
import time
import numpy as np
from PIL import Image
from typing import Tuple
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import (
    EfficientViTSamAutomaticMaskGenerator,
)

# -----------------------------
# 사용자 경로/파라미터
# -----------------------------
IMG_DIR = "/workspace/gahyeon/dataset/garden-original-clear-mask/images"
CKPT    = "/workspace/gahyeon/efficientvit_sam/weight/efficientvit_sam_xl1.pt"
OUT_DIR = "/workspace/gahyeon/dataset/garden-original-clear-mask/images/m"
TEXT    = "a photo of a table, a dining table, a wooden table, a photo of a table"
MAX_SHORT_SIDE = 1024
POINTS_PER_SIDE = 16
CROP_N_LAYERS = 0
THRESH = 0.5
MORPH_K = 5

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

# -----------------------------
# 개별 이미지 처리 함수
# -----------------------------
def process_one_image(img_path, out_path, sam, clip_model, preprocess, text_tok, device):
    start_time = time.time()  # 시작 시간 기록

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Cannot read image: {img_path}")
        return None
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_small, scale = resize_keep_aspect(img_rgb, MAX_SHORT_SIDE)

    amg = EfficientViTSamAutomaticMaskGenerator(
        sam,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=32,
        crop_n_layers=CROP_N_LAYERS,
        crop_n_points_downscale_factor=2,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        box_nms_thresh=0.7,
    )
    masks_small = amg.generate(img_small)

    if len(masks_small) == 0:
        print(f"❌ No masks produced for {img_path}")
        return None

    best_score = -1e9
    best_mask_full = None

    for m in masks_small:
        m_small = m["segmentation"].astype(np.uint8)
        m_full = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
        m_u8 = (m_full * 255).astype(np.uint8)

        masked = cv2.bitwise_and(img_rgb, img_rgb, mask=m_u8)
        pil = Image.fromarray(masked)
        clip_in = preprocess(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = clip_model.encode_image(clip_in)
            txt_feat = clip_model.encode_text(text_tok)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            score = float((img_feat @ txt_feat.T).item())

        if score > best_score:
            best_score = score
            best_mask_full = m_u8

    if best_mask_full is None:
        print(f"❌ No mask selected for {img_path}")
        return None

    best_mask_full = ((best_mask_full > (THRESH * 255)).astype(np.uint8) * 255)
    best_mask_full = morph_cleanup(best_mask_full, k=MORPH_K)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, best_mask_full)

    elapsed = time.time() - start_time
    print(f"✅ Saved {out_path} (CLIP score {best_score:.4f}, time {elapsed:.2f}s)")
    return elapsed

# -----------------------------
# 메인
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=False)
    sam.load_state_dict(torch.load(CKPT, map_location=device))
    sam = sam.to(device).eval()

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_tok = clip.tokenize([TEXT]).to(device)

    os.makedirs(OUT_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    total_time = 0
    count = 0
    for fname in img_files:
        img_path = os.path.join(IMG_DIR, fname)
        out_path = os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}_bike_mask.png")
        print(f"🔹 Processing {img_path}")
        elapsed = process_one_image(img_path, out_path, sam, clip_model, preprocess, text_tok, device)
        if elapsed is not None:
            total_time += elapsed
            count += 1

    if count > 0:
        print(f"\n📊 Average time per image: {total_time / count:.2f}s over {count} images")

if __name__ == "__main__":
    main()
