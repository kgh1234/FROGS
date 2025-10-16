import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator

# -----------------------------
# 사용자 경로/파라미터
# -----------------------------
IMG_DIR = "/workspace/gahyeon/dataset/garden/images"
CKPT    = "/workspace/gahyeon/efficientvit_sam/weight/efficientvit_sam_xl1.pt"
OUT_DIR = "/workspace/gahyeon/efficientvit_sam/outputs/table_mask_time"
TEXT    = "a photo of a table, a dining table, a wooden table, a photo of a table"

MAX_SHORT_SIDE   = 1024
# AMG 튜닝: 후보 풍부하게(시간/메모리 허용 시 점차 올려보기)
POINTS_PER_SIDE  = 32          # 16→32 권장 (64는 메모리/시간↑)
CROP_N_LAYERS    = 1           # 0→1 권장 (더 많은 후보)
# 후보 필터를 너무 빡빡하지 않게: CLIP이 걸러줌
PRED_IOU_THR     = 0.86
STAB_SCORE_THR   = 0.90
BOX_NMS_THR      = 0.7

# 후처리
MORPH_K          = 5           # 클로징/오프닝 커널
MIN_AREA_RATIO   = 0.002       # 최종 최소 면적 비율(전체 픽셀 대비 0.2% 미만이면 보정)
DILATE_PX        = 2           # 살짝 팽창해 경계 보강
REMOVE_SMALL_PX  = 200         # 작은 섬 제거(픽셀 수 기준)
UNION_TAU        = 0.95        # 최상위 점수의 95% 이상 후보는 union

# 점수 가중치 (CLIP 중심 + SAM의 품질신호 + 면적)
W_CLIP, W_IOU, W_STAB, W_AREA = 0.6, 0.2, 0.1, 0.1

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

def remove_small_components(mask_u8: np.ndarray, min_px: int) -> np.ndarray:
    """연결요소 중 작은 섬 제거"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[labels == i] = 255
    return keep

def dilate(mask_u8: np.ndarray, px: int) -> np.ndarray:
    if px <= 0: return mask_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.dilate(mask_u8, k, iterations=1)

def masked_clip_score(img_rgb: np.ndarray, mask_u8: np.ndarray, clip_model, preprocess, txt_feat, device) -> float:
    # 배경은 검정으로, 객체만 남긴 버전으로 CLIP
    masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_u8)
    pil = Image.fromarray(masked)
    clip_in = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(clip_in)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        score = float((img_feat @ txt_feat.T).item())
    return score

def pick_and_refine_mask(img_rgb: np.ndarray,
                         masks_small: List[Dict[str, Any]],
                         H: int, W: int,
                         clip_model, preprocess, txt_feat, device) -> np.ndarray:
    """CLIP+SAM 품질+면적 가중합으로 상위 후보 선택, 근접 후보 union 후 후처리"""
    if len(masks_small) == 0:
        return None

    # 각 후보 점수 계산
    scores = []
    u8_masks_full = []
    areas = []
    for m in masks_small:
        m_small = m["segmentation"].astype(np.uint8)          # {0,1}
        m_full  = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
        m_u8    = (m_full * 255).astype(np.uint8)
        area    = (m_u8.sum() / 255.0)

        # CLIP
        s_clip  = masked_clip_score(img_rgb, m_u8, clip_model, preprocess, txt_feat, device)
        # SAM 메타
        s_iou   = float(m.get("predicted_iou", 0.0))
        s_stab  = float(m.get("stability_score", 0.0))
        # 면적 정규화
        area_n  = area / float(H*W)

        s_total = (W_CLIP * s_clip) + (W_IOU * s_iou) + (W_STAB * s_stab) + (W_AREA * area_n)

        scores.append(s_total)
        u8_masks_full.append(m_u8)
        areas.append(area)

    scores = np.array(scores, dtype=np.float32)
    best_idx = int(scores.argmax())
    best = u8_masks_full[best_idx]

    # 최상위에 근접한 후보는 union으로 살짝 확장(경계/얇은 구조 보강)
    thresh = UNION_TAU * float(scores.max())
    union_mask = best.copy()
    for m_u8, s in zip(u8_masks_full, scores):
        if s >= thresh:
            union_mask = cv2.bitwise_or(union_mask, m_u8)

    # 후처리: 작은 섬 제거 → 클로징/오프닝 → 소폭 팽창
    union_mask = remove_small_components(union_mask, REMOVE_SMALL_PX)
    union_mask = morph_cleanup(union_mask, k=MORPH_K)
    union_mask = dilate(union_mask, px=DILATE_PX)

    # 최소 면적 보장(언더-세그 방지)
    if (union_mask.sum() / 255.0) < (MIN_AREA_RATIO * H * W):
        # 너무 작으면 '조금 더 팽창' 한 번 더
        union_mask = dilate(union_mask, px=max(2, DILATE_PX + 2))

    return union_mask

# -----------------------------
# 개별 이미지 처리
# -----------------------------
def process_one_image(img_path, out_path, sam, clip_model, preprocess, txt_feat, device):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Cannot read image: {img_path}")
        return
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_small, _ = resize_keep_aspect(img_rgb, MAX_SHORT_SIDE)

    amg = EfficientViTSamAutomaticMaskGenerator(
        sam,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=32,
        crop_n_layers=CROP_N_LAYERS,
        crop_n_points_downscale_factor=2,
        pred_iou_thresh=PRED_IOU_THR,
        stability_score_thresh=STAB_SCORE_THR,
        box_nms_thresh=BOX_NMS_THR,
    )
    masks_small = amg.generate(img_small)

    # 후보 없으면 "빈(검정) 마스크"라도 저장해 다운스트림 파이프라인 깨지지 않게
    if len(masks_small) == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, np.zeros((H, W), np.uint8))
        print(f"⚠️ No masks produced for {img_path}. Saved empty mask: {out_path}")
        return

    best_mask = pick_and_refine_mask(img_rgb, masks_small, H, W, clip_model, preprocess, txt_feat, device)
    if best_mask is None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, np.zeros((H, W), np.uint8))
        print(f"⚠️ Selection failed for {img_path}. Saved empty mask: {out_path}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, best_mask)
    print(f"✅ Saved {out_path}")

# -----------------------------
# 메인
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=False)
    sam.load_state_dict(torch.load(CKPT, map_location=device))
    sam = sam.to(device).eval()

    # CLIP은 한 번만 준비(속도↑). 메모리 여유 있으면 "ViT-L/14" 써도 품질↑(속도↓).
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        text_tok = clip.tokenize([TEXT]).to(device)
        txt_feat = clip_model.encode_text(text_tok)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for fname in img_files:
        img_path = os.path.join(IMG_DIR, fname)
        out_path = os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}_table_mask.png")
        print(f"🔹 Processing {img_path}")
        process_one_image(img_path, out_path, sam, clip_model, preprocess, txt_feat, device)

if __name__ == "__main__":
    main()
