#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
# =============================================

SCENE_NAME="lerf_mask"
ROOT="../../masked_datasets/$SCENE_NAME"
OUTPUT_ROOT="../../output_pruning/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_$SCENE_NAME.csv"
SHEET_NAME="Ours_Original_Masked_3DGS"


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

    fi
        # 3) Training / Rendering / Metrics
        echo "Training 시작..."
        if [ -d "$OUT_DIR" ]; then
            echo "'$OUT_DIR' 이미 존재 — 건너뜀"
            continue
        fi
        python train.py -s "$SCENE_PATH" -m "$OUT_DIR" --eval

        echo "Rendering: $SCENE"
        python render.py -m "$OUT_DIR" 

        echo "Evaluating metrics: $SCENE"
        python metrics.py -m "$OUT_DIR" | tee metrics_tmp.log

        # metrics 값 추출
        SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
        PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
        LPIPS=$(grep "LPIPS" metrics_tmp.log | awk '{print $2}')

        # CSV 작성
        if [ ! -f "$CSV_FILE" ]; then
            echo "scene,SSIM,PSNR,LPIPS" > "$CSV_FILE"
        fi
        echo "$SCENE,$SSIM,$PSNR,$LPIPS" >> "$CSV_FILE"

        echo "Metrics for $SCENE appended to $CSV_FILE"
        echo "Finished: $SCENE"
        echo
    fi
done

echo "All scenes processed."
