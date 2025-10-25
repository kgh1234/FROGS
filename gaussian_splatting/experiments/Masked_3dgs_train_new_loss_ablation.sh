#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT, with alpha ablation
# =============================================

SCENE_NAME="lerf_mask"
ROOT="../masked_datasets/$SCENE_NAME"
OUTPUT_ROOT="../ablation_inoutloss/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_$SCENE_NAME.csv"
SHEET_NAME="ablation_inoutloss"

# 여러 alpha 값 반복
ALPHAS=(0 0.00001 0.001 0.01 0.1 0.5 1.0)

export CUDA_VISIBLE_DEVICES=0

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")
        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/mask"
        ORI_DIR="$SCENE_PATH/images_ori"

        for ALPHA in "${ALPHAS[@]}"; do
            OUT_DIR="$OUTPUT_ROOT/${SCENE}_alpha${ALPHA}"

            echo "====================================="
            echo "Processing scene: $SCENE | alpha=$ALPHA"
            echo "====================================="

            echo " Training 시작..."
            TRAIN_START=$(date +%s)
            LOGFILE="vram_${SCENE}_alpha${ALPHA}.log"
            nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -l 2 > "$LOGFILE" &
            VRAM_PID=$!

            python train_inoutloss.py -s "$SCENE_PATH" -m "$OUT_DIR" --mask_dir "$MASK_DIR" --alpha $ALPHA --eval

            TRAIN_END=$(date +%s)
            TRAIN_TIME=$((TRAIN_END - TRAIN_START))
            echo "Training time: ${TRAIN_TIME}s"

            kill $VRAM_PID 2>/dev/null
            VRAM_MAX=$(awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "$LOGFILE")
            rm -f "$LOGFILE"

            echo "Rendering: $SCENE (alpha=$ALPHA)"
            RENDER_START=$(date +%s)
            python render.py -m "$OUT_DIR"
            RENDER_END=$(date +%s)
            RENDER_TIME=$((RENDER_END - RENDER_START))
            echo "Rendering time: ${RENDER_TIME}s"

            echo "Evaluating metrics: $SCENE (alpha=$ALPHA)"
            python metrics_object.py -m "$OUT_DIR" --mask_dir "$MASK_DIR" | tee metrics_tmp.log

            POINT_CLOUD_DIR="$OUT_DIR/point_cloud"
            if [ -d "$POINT_CLOUD_DIR" ]; then
                LATEST_ITER_DIR=$(ls -d "$POINT_CLOUD_DIR"/iteration_* 2>/dev/null | sort -V | tail -n 1)
                if [ -n "$LATEST_ITER_DIR" ]; then
                    PLY_PATH="$LATEST_ITER_DIR/point_cloud.ply"
                    if [ -f "$PLY_PATH" ]; then
                        GAUSSIAN_COUNT=$(grep -a -m1 "element vertex" "$PLY_PATH" | awk '{print $3}')
                        echo "Gaussian: $GAUSSIAN_COUNT (from $(basename "$LATEST_ITER_DIR"))"
                    else
                        echo "Gaussian: PLY not found in $(basename "$LATEST_ITER_DIR")"
                    fi
                else
                    echo "Gaussian: No iteration_* folder found under $POINT_CLOUD_DIR"
                fi
            else
                echo "Gaussian: point_cloud folder not found in $OUT_DIR"
            fi

            # metrics 추출
            SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
            PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
            LPIPS=$(grep -oP 'LPIPS\s*:\s*\K[0-9.e+-]+' metrics_tmp.log)

            # Google Sheet 업데이트
            python ../../update_sheet.py "$SHEET_NAME" "${SCENE}_alpha${ALPHA}" "$SSIM" "$PSNR" "$LPIPS" "$TRAIN_TIME" "$RENDER_TIME" "$VRAM_MAX" "$GAUSSIAN_COUNT"

            # CSV 작성
            if [ ! -f "$CSV_FILE" ]; then
                echo "scene,alpha,SSIM,PSNR,LPIPS,TRAIN_TIME,RENDER_TIME,VRAM_MAX,GAUSSIAN_COUNT" > "$CSV_FILE"
            fi
            echo "$SCENE,$ALPHA,$SSIM,$PSNR,$LPIPS,$TRAIN_TIME,$RENDER_TIME,$VRAM_MAX,$GAUSSIAN_COUNT" >> "$CSV_FILE"

            echo "Metrics for $SCENE (alpha=$ALPHA) appended to $CSV_FILE"
            echo "Finished: $SCENE (alpha=$ALPHA)"
            echo
        done
    fi
done

echo "All scenes & alphas processed."
