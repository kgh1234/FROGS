import os
import time
import numpy as np
import torch
import cv2
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor


# ====== PATHS ======
CONFIG_FILE = "/workspace/gahyeon/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "/workspace/gahyeon/GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAM_CKPT = "/workspace/gahyeon/efficientvit_sam/weight/efficientvit_sam_xl1.pt"

IMAGE_DIR = "/workspace/gahyeon/dataset/bicycle/images"
OUTPUT_DIR = "/workspace/gahyeon/dataset/bicycle-mask/DINO_refined_v2"

CPU_ONLY = False

# ====== 한 문장 프롬프트 앙상블 (각 항목은 반드시 한 문장) ======
PROMPT_CANDIDATES = [
    "a bicycle standing next to a black bench in a park",
    "a white bicycle parked by a bench",
    "a bike leaning against a park bench",
    "a white road bike in the grass near a bench",
    "a bicycle in a park scene"
]

# 임계치 스윕 (앞에서부터 시도)
BOX_TH_STEPS  = [0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.27, 0.25]
TEXT_TH_STEPS = [0.35, 0.30, 0.27, 0.25, 0.22, 0.20]

TOPK_BOXES = 10        # 상위 박스 후보 개수
NMS_IOU_TH = 0.5       # NMS IoU 임계치
TINY_AREA_RATIO = 0.0002  # (H*W)*ratio 보다 작으면 잡음으로 취급


def _normalize_caption(caption: str) -> str:
    caption = caption.strip()
    if not caption.endswith("."):
        caption += "."
    return caption.lower()


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),  # 사실상 고정 리사이즈
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if (torch.cuda.is_available() and not cpu_only) else "cpu"
    model = build_model(args)
    ckpt = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    print("GroundingDINO load:", load_res)
    model.eval()
    return model


def xywhn_to_xyxy(box, W, H):
    """cx,cy,w,h (norm) -> x0,y0,x1,y1 (pixels) + 경계 클램프"""
    box = box * torch.tensor([W, H, W, H])
    cx, cy, w, h = box
    x0 = float(cx - w / 2.0)
    y0 = float(cy - h / 2.0)
    x1 = float(cx + w / 2.0)
    y1 = float(cy + h / 2.0)
    x0 = int(max(0, min(W - 1, x0)))
    y0 = int(max(0, min(H - 1, y0)))
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    if x1 <= x0: x1 = min(W - 1, x0 + 1)
    if y1 <= y0: y1 = min(H - 1, y0 + 1)
    return np.array([x0, y0, x1, y1], dtype=np.int32)


def box_area_xyxy(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def iou_xyxy(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1-x0) * max(0, y1-y0)
    if inter == 0: return 0.0
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return inter / max(union, 1e-6)


def nms_xyxy(boxes, scores, iou_th):
    keep = []
    idxs = np.argsort(-scores)  # desc
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1: break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious < iou_th]
    return keep


@torch.no_grad()
def run_dino_once(model, image_tensor, caption, box_th, text_th, cpu_only=False):
    """한 문장 캡션/임계치로 1회 실행 -> 필터링/정렬된 박스/문구/점수 반환"""
    device = "cuda" if (torch.cuda.is_available() and not cpu_only) else "cpu"
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    caption = _normalize_caption(caption)

    outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (Q, V)
    boxes  = outputs["pred_boxes"][0]             # (Q, 4) normalized

    scores = logits.max(dim=1)[0]                 # (Q,)
    keep = scores > box_th
    if keep.sum().item() == 0:
        return [], [], [], caption

    logits_f = logits[keep].cpu()
    boxes_f  = boxes[keep].cpu()
    scores_f = scores[keep].cpu()

    # 문구 (디버깅용)
    tokenizer = model.tokenizer
    tok = tokenizer(caption)
    phrases = []
    for logit in logits_f:
        phrases.append(get_phrases_from_posmap(logit > text_th, tok, tokenizer))

    # 점수 기반 정렬 후 TOP-K
    order = torch.argsort(scores_f, descending=True)
    boxes_f = boxes_f[order][:TOPK_BOXES]
    phrases = [phrases[i] for i in order.tolist()[:TOPK_BOXES]]
    scores_f = scores_f[order][:TOPK_BOXES]

    return boxes_f, phrases, scores_f, caption


def pick_best_mask(masks, H, W):
    """
    멀티마스크 중 '면적(우선) + 화면 중심에 가까움(보조)'으로 최고 마스크 선택
    """
    if masks.ndim == 3:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)  # 픽셀 수
        # 중심성 점수 (마스크의 평균 좌표가 중앙에 가까울수록 좋음)
        ys, xs = np.indices((H, W))
        scores = []
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.float32)
            denom = m.sum() + 1e-6
            cy = (ys * m).sum() / denom
            cx = (xs * m).sum() / denom
            # 화면 중앙 (H/2, W/2)와의 거리
            dist = np.sqrt((cy - H/2)**2 + (cx - W/2)**2)
            scores.append((-dist, areas[i]))  # 중앙에 가까울수록 dist 작음 → 점수 큼
        idx = int(np.lexsort((np.array(scores)[:,1], np.array(scores)[:,0]))[-1])  # (-dist, area) 기준
        return masks[idx]
    return masks


def draw_overlay(image_bgr, boxes_xyxy, best_mask=None):
    vis = image_bgr.copy()
    # 박스
    for b in boxes_xyxy:
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    # 마스크
    if best_mask is not None:
        colored = np.zeros_like(vis)
        colored[:, :, 1] = (best_mask.astype(np.uint8) * 180)  # G 채널
        vis = cv2.addWeighted(vis, 1.0, colored, 0.4, 0)
    return vis


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if (torch.cuda.is_available() and not CPU_ONLY) else "cpu"
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    dino = load_model(CONFIG_FILE, CHECKPOINT_PATH, cpu_only=CPU_ONLY)
    sam = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=False)
    sam.load_state_dict(torch.load(SAM_CKPT, map_location=device))
    sam = sam.to(device).eval()
    predictor = EfficientViTSamPredictor(sam)

    img_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    total_time, cnt = 0.0, 0
    for fname in img_files:
        img_path = os.path.join(IMAGE_DIR, fname)
        out_mask_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}_mask.png")
        out_overlay_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}_overlay.jpg")
        print(f"🔹 Processing {img_path}")

        t0 = time.time()
        pil_img, img_tensor = load_image(img_path)
        img_np = np.array(pil_img)   # HWC RGB
        H, W = img_np.shape[:2]
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        found = False
        best_mask_to_save = None
        best_overlay_boxes = None

        # 프롬프트 × 임계치 스윕
        for prompt in PROMPT_CANDIDATES:
            for box_th in BOX_TH_STEPS:
                for text_th in TEXT_TH_STEPS:
                    boxes_f, phrases_f, scores_f, used_caption = run_dino_once(
                        dino, img_tensor, prompt, box_th, text_th, cpu_only=CPU_ONLY
                    )
                    if len(boxes_f) == 0:
                        continue

                    # 정규화 -> 픽셀 박스 변환
                    boxes_xyxy = [xywhn_to_xyxy(b, W, H) for b in boxes_f]
                    scores_np = np.array([float(s) for s in scores_f])

                    # NMS
                    keep_idx = nms_xyxy(boxes_xyxy, scores_np, NMS_IOU_TH)
                    boxes_xyxy = [boxes_xyxy[i] for i in keep_idx]
                    scores_np = scores_np[keep_idx]

                    if len(boxes_xyxy) == 0:
                        continue

                    # SAM 예측
                    predictor.set_image(img_np)
                    saved = False
                    # 점수 내림차순
                    order = np.argsort(-scores_np)
                    for j in order:
                        b = boxes_xyxy[j]
                        masks, _, _ = predictor.predict(box=b, multimask_output=True)
                        mask = pick_best_mask(masks, H, W)  # (H, W) bool
                        mask_u8 = (mask.astype(np.uint8)) * 255

                        # 너무 작은 마스크 필터링
                        if mask_u8.sum() < TINY_AREA_RATIO * (H * W) * 255:
                            continue

                        # 저장 & 오버레이
                        cv2.imwrite(out_mask_path, mask_u8)
                        overlay = draw_overlay(img_bgr, [b], best_mask=mask)
                        cv2.imwrite(out_overlay_path, overlay)

                        print(f"  ✅ Saved mask: {out_mask_path}")
                        print(f"     box={b} score={scores_np[j]:.3f} prompt='{used_caption}' box_th={box_th} text_th={text_th}")
                        best_mask_to_save = mask_u8
                        best_overlay_boxes = [b]
                        saved = True
                        found = True
                        break

                    if found:
                        break
                if found:
                    break
            if found:
                break

        if not found:
            # 완전 실패 → 블랙 마스크 저장 + 빈 오버레이(박스x)
            black = np.zeros((H, W), dtype=np.uint8)
            cv2.imwrite(out_mask_path, black)
            vis = draw_overlay(img_bgr, [])
            cv2.imwrite(out_overlay_path, vis)
            print(f"  ❌ No detection. Saved black mask: {out_mask_path}")

        total_time += (time.time() - t0)
        cnt += 1

    if cnt > 0:
        print(f"\n📊 Average time per image: {total_time / cnt:.2f}s over {cnt} images")


if __name__ == "__main__":
    main()
