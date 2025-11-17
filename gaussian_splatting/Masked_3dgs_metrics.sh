#!/bin/bash
# =============================================
# 3DGS Metrics Logging for multiple runs
# Each run (run1, run2, run3) is appended as a row
# =============================================

SCENE_NAME="mipnerf"
ROOT="../../masked_datasets/$SCENE_NAME"
OUTPUT_ROOT="../../output_mini/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_${SCENE_NAME}.csv"
SHEET_NAME="mini"

export CUDA_VISIBLE_DEVICES=0

# CSV 헤더 (없으면 생성)
if [ ! -f "$CSV_FILE" ]; then
    echo "scene,SSIM,PSNR,LPIPS" > "$CSV_FILE"
fi

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")
        MASK_DIR="$SCENE_PATH/mask"
        SCENE_NAME="${SCENE%%_*}"

        # run1~run3 반복
        for RUN_ID in 1 2 3; do
            OUT_DIR="$OUTPUT_ROOT/${SCENE_NAME}_run${RUN_ID}"

            if [ ! -d "$OUT_DIR" ]; then
                echo "⚠️  Skipping $OUT_DIR (not found)"
                continue
            fi

            echo "🔍 Evaluating $SCENE_NAME (run${RUN_ID})"
            python metrics_object.py -m "$OUT_DIR" -mask "$MASK_DIR" | tee metrics_tmp.log

            # metrics 값 추출
            SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
            PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
            LPIPS=$(grep "LPIPS" metrics_tmp.log | awk '{print $2}')

            # CSV 작성 (run별로 한 줄씩)
            echo "${SCENE_NAME}_run${RUN_ID},$SSIM,$PSNR,$LPIPS" >> "$CSV_FILE"

            # Google Sheet 업데이트
            python ../../update_sheet.py "$SHEET_NAME" "${SCENE_NAME}_run${RUN_ID}" "$SSIM" "$PSNR" "$LPIPS" "" "" "" "" ""

            echo "Recorded: ${SCENE_NAME}_run${RUN_ID}"
            echo
        done
    fi
done

echo "All scenes processed. CSV saved to $CSV_FILE"
