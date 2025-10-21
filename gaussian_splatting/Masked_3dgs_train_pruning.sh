#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
# =============================================

SCENE_NAME="lerf_mask_ours"
ROOT="../dataset/$SCENE_NAME"
OUTPUT_ROOT="output_pruning/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_$SCENE_NAME.csv"

export CUDA_VISIBLE_DEVICES=0

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")
        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/mask"
        ORI_DIR="$SCENE_PATH/images_ori"
        OUT_DIR="$OUTPUT_ROOT/${SCENE}"

        echo "====================================="
        echo "Processing scene: $SCENE"
        echo "====================================="

        # 1) 이미지 백업
        if [ ! -d "$ORI_DIR" ]; then
            echo "[1/3] 백업 및 폴더 준비: $ORI_DIR"
            if [ -d "$IMG_DIR" ] && [ ! -d "$ORI_DIR" ]; then
                mv "$IMG_DIR" "$ORI_DIR"
                echo "'$IMG_DIR' → '$ORI_DIR' 로 이동 완료"
            else
                echo "이미 '$ORI_DIR' 존재하거나 '$IMG_DIR' 없음 — 건너뜀"
            fi
            mkdir -p "$IMG_DIR"

            # 2) 마스크 적용
            echo "[2/3] 마스크 적용 중..."
            python3 - <<PYCODE
import os, cv2, numpy as np
from pathlib import Path

scene_dir = Path("$SCENE_PATH")
ori_dir = scene_dir / "images_ori"
mask_dir = scene_dir / "mask"
out_dir = scene_dir / "images"
out_dir.mkdir(parents=True, exist_ok=True)

for fname in sorted(os.listdir(ori_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = ori_dir / fname
    out_path = out_dir / fname

    base = Path(fname).stem
    mask_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        p = mask_dir / f"{base}{ext}"
        if p.exists():
            mask_path = p
        break
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"이미지 읽기 실패: {fname}")
        continue

    if mask is None:
        print(f"마스크 없음 → 원본 복사: {fname}")
        cv2.imwrite(str(out_path), img)
        continue

    mask_bin = (mask > 127).astype(np.uint8)
    masked = img * mask_bin[:, :, None]
    cv2.imwrite(str(out_path), masked)
    #print(f"마스크 적용 완료: {fname}")

print("모든 이미지 마스크 적용 완료 → 'images/' 폴더 저장됨")
PYCODE
    fi
        # 3) Training / Rendering / Metrics
        echo "[3/3] Training 시작..."
        if [ -d "$OUT_DIR" ]; then
            echo "'$OUT_DIR' 이미 존재 — 건너뜀"
            #continue
        fi
        python train_pruning.py -s "$SCENE_PATH" -m "$OUT_DIR" --mask_dir "$MASK_DIR" --eval

        echo "Rendering: $SCENE"
        python render.py -m "$OUT_DIR" 

        # echo "Evaluating metrics: $SCENE"
        # python metrics.py -m "$OUT_DIR" | tee metrics_tmp.log

        # # metrics 값 추출
        # SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
        # PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
        # LPIPS=$(grep -oP 'LPIPS\s*:\s*\K[0-9.e+-]+' metrics_tmp.log)

        # # CSV 작성
        # if [ ! -f "$CSV_FILE" ]; then
        #     echo "scene,SSIM,PSNR,LPIPS" > "$CSV_FILE"
        # fi
        # echo "$SCENE,$SSIM,$PSNR,$LPIPS" >> "$CSV_FILE"

        # echo "Metrics for $SCENE appended to $CSV_FILE"
        # echo "Finished: $SCENE"
        # echo
    fi
done

echo "All scenes processed."
