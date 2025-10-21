#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
# =============================================

SCENE_NAME="mipnes"
ROOT="../../original_datasets/$SCENE_NAME"
OUTPUT_ROOT="../../output_original/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_$SCENE_NAME.csv"
SHEET_NAME="Original_3DGS"

export CUDA_VISIBLE_DEVICES=0

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")
        IMG_DIR="$SCENE_PATH/images"
        ORI_DIR="$SCENE_PATH/images_ori"
        OUT_DIR="$OUTPUT_ROOT/${SCENE}"

        echo "====================================="
        echo "Processing scene: $SCENE"
        echo "====================================="

        echo "[1/1] Training 시작: $SCENE"

        TRAIN_START=$(date +%s)
        LOGFILE="vram_${SCENE}.log"
        nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -l 2 > "$LOGFILE" &
        VRAM_PID=$!

        python train.py -s "$SCENE_PATH" -m "$OUT_DIR" --eval

        TRAIN_END=$(date +%s)
        TRAIN_TIME=$((TRAIN_END - TRAIN_START))
        echo "Training time: ${TRAIN_TIME}s"

        kill $VRAM_PID 2>/dev/null
        VRAM_MAX=$(awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "$LOGFILE")
        rm -f "$LOGFILE"

        echo "Rendering: $SCENE"
        RENDER_START=$(date +%s)
        python render.py -m "$OUT_DIR" | tee render_tmp.log
        RENDER_END=$(date +%s)
        RENDER_TIME=$((RENDER_END - RENDER_START))
        echo "Rendering time: ${RENDER_TIME}s"


        echo "Evaluating metrics: $SCENE"
        python metrics.py -m "$OUT_DIR" | tee metrics_tmp.log

        SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
        PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
        LPIPS=$(grep "LPIPS" metrics_tmp.log | awk '{print $2}')

        python ../../update_sheet.py "$SHEET_NAME" "$SCENE" "$SSIM" "$PSNR" "$LPIPS" "$TRAIN_TIME" "$RENDER_TIME" "$VRAM_MAX"

        # CSV 작성
        if [ ! -f "$CSV_FILE" ]; then
            echo "scene,SSIM,PSNR,LPIPS,TrainTime(s),RenderTime(s),MaxVRAM(MB)" > "$CSV_FILE"
        fi
        echo "$SCENE,$SSIM,$PSNR,$LPIPS,$TRAIN_TIME,$RENDER_TIME,$VRAM_MAX" >> "$CSV_FILE"

    echo "Metrics for $SCENE appended to $CSV_FILE and Google Sheet"
    echo "Finished: $SCENE"
    fi
done

echo "All scenes processed."
