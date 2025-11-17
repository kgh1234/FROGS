
import os
import time
import json
import numpy as np
import torch
import cv2
from PIL import Image
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

# ====== PATHS ======
DTU_ROOT = "../../masked_datasets/tanks_temples"
SAM_CKPT = "./weight/efficientvit_sam_xl1.pt"

DATASET_ROOT = DTU_ROOT
OUTPUT_ROOT = DTU_ROOT
CPU_ONLY = False
TINY_AREA_RATIO = 0.0002


def pick_best_mask(masks, H, W):
    """멀티마스크 중 중심성 + 면적 기준으로 최적 마스크 선택"""
    if masks.ndim == 3:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        ys, xs = np.indices((H, W))
        scores = []
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.float32)
            denom = m.sum() + 1e-6
            cy = (ys * m).sum() / denom
            cx = (xs * m).sum() / denom
            dist = np.sqrt((cy - H/2)**2 + (cx - W/2)**2)
            scores.append((-dist, areas[i]))
        idx = int(np.lexsort((np.array(scores)[:, 1], np.array(scores)[:, 0]))[-1])
        return masks[idx]
    return masks


def draw_overlay(image_bgr, boxes_xyxy, best_mask=None):
    vis = image_bgr.copy()
    for b in boxes_xyxy:
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    if best_mask is not None:
        colored = np.zeros_like(vis)
        colored[:, :, 1] = (best_mask.astype(np.uint8) * 180)
        vis = cv2.addWeighted(vis, 1.0, colored, 0.4, 0)
    return vis


def process_scan(scan_name, predictor):
    """각 scan 폴더의 bbox JSON을 불러와 SAM으로 마스크 생성"""
    scan_path = os.path.join(DATASET_ROOT, scan_name)
    image_dir = os.path.join(scan_path, "images")
    label_dir = os.path.join(scan_path, "labels")
    output_dir = os.path.join(OUTPUT_ROOT, scan_name, "masks")

    if not os.path.exists(image_dir):
        print(f"[WARN] {scan_name}: images 폴더 없음. 건너뜀.")
        return
    if not os.path.exists(label_dir):
        print(f"[WARN] {scan_name}: labels 폴더 없음. 건너뜀.")
        return

    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[{scan_name}] {len(img_files)}개 이미지 처리 중...")

    total_time, cnt = 0.0, 0

    for fname in img_files:
        name, _ = os.path.splitext(fname)
        json_path = os.path.join(label_dir, f"{name}_bbox.json")
        img_path = os.path.join(image_dir, fname)
        out_mask_path = os.path.join(output_dir, f"{name}_mask.png")
        # out_overlay_path = os.path.join(output_dir, f"{name}_overlay.jpg")

        if os.path.exists(out_mask_path):
            print(f"⏭️ {fname}: 이미 마스크 존재 → 건너뜀.")
            continue
        
        if not os.path.exists(json_path):
            print(f"⚠️ {fname}: bbox json 없음 → 건너뜀.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bboxes = data.get("bboxes", [])
        if not bboxes:
            print(f"⚠️ {fname}: bbox 정보 없음 → 건너뜀.")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"❌ 이미지를 읽을 수 없음: {img_path}")
            continue

        H, W = img_bgr.shape[:2]
        predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        found = False
        t0 = time.time()

        # bbox 리스트 반복 (하나만 있어도 동작)
        for box in bboxes:
            b = np.array([box["x1"], box["y1"], box["x2"], box["y2"]])
            masks, _, _ = predictor.predict(box=b, multimask_output=True)
            mask = pick_best_mask(masks, H, W)
            mask_u8 = (mask.astype(np.uint8)) * 255

            # 너무 작은 마스크는 무시
            if mask_u8.sum() < TINY_AREA_RATIO * (H * W) * 255:
                continue

            cv2.imwrite(out_mask_path, mask_u8)
            overlay = draw_overlay(img_bgr, [b], best_mask=mask)
            # cv2.imwrite(out_overlay_path, overlay)
            print(f"✅ {scan_name}: Saved mask for {fname}")
            found = True
            break

        if not found:
            black = np.zeros((H, W), dtype=np.uint8)
            cv2.imwrite(out_mask_path, black)
            print(f"❌ {scan_name}: No valid mask for {fname}")

        total_time += time.time() - t0
        cnt += 1

    print(f"[{scan_name}] 평균 처리 시간: {total_time / max(cnt, 1):.2f}s")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = "cuda" if (torch.cuda.is_available() and not CPU_ONLY) else "cpu"
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    print("🧠 SAM 모델 로드 중...")
    sam = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=False)
    sam.load_state_dict(torch.load(SAM_CKPT, map_location=device))
    sam = sam.to(device).eval()
    predictor = EfficientViTSamPredictor(sam)
    print("✅ SAM 로드 완료\n")

    # target=['Barn', 'Truck']

    # 모든 scan 폴더 순회
    for scan_name in sorted(os.listdir(DATASET_ROOT)):
        # if scan_name not in target:
        #     continue
        if not os.path.isdir(os.path.join(DATASET_ROOT, scan_name)):
            continue
        process_scan(scan_name, predictor)

    print("\n🎉 모든 scan 처리 완료!")


if __name__ == "__main__":
    main()
