#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
# =============================================

SCENE_NAME="lerf_mask_refine"
ROOT="../../FROGS_input_dataset/masked_datasets/$SCENE_NAME"
OUTPUT_ROOT="../../output_prune_ratio/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_$SCENE_NAME.csv"
SHEET_NAME="ablation_pruneratio"


export CUDA_VISIBLE_DEVICES=0

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        #SCENE_PATH="../../FROGS_input_dataset/masked_datasets/$SCENE_NAME/figurines_15"
        SCENE=$(basename "$SCENE_PATH")

        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/mask_refine2"
        ORI_DIR="$SCENE_PATH/images_ori"
        OUT_DIR="$OUTPUT_ROOT/${SCENE}/$(date -d '+9 hours' +%m%d_%H%M)_prune0.4"


        echo "====================================="
        echo "Processing scene: $SCENE"
        echo "====================================="

        echo " Training 시작..."
        
        TRAIN_START=$(date +%s)
        LOGFILE="vram_${SCENE}.log"
        nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -l 2 > "$LOGFILE" &
        VRAM_PID=$!

        python train_all.py -s "$SCENE_PATH" -m "$OUT_DIR" --mask_dir "$MASK_DIR" --prune_iterations 0 --prune_ratio 0.4 --eval 

        TRAIN_END=$(date +%s)
        TRAIN_TIME=$((TRAIN_END - TRAIN_START))
        echo "Training time: ${TRAIN_TIME}s"

        kill $VRAM_PID 2>/dev/null
        VRAM_MAX=$(awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "$LOGFILE")
        rm -f "$LOGFILE"

        echo "Rendering: $SCENE"
 
        RENDER_START=$(date +%s)
        python render_with_mask.py -m "$OUT_DIR"
        RENDER_END=$(date +%s)
        RENDER_TIME=$((RENDER_END - RENDER_START))
        echo "Rendering time: ${RENDER_TIME}s"



        echo "Evaluating metrics: $SCENE"
        python metrics_object_mIoU.py -m "$OUT_DIR" --mask_dir "$MASK_DIR" | tee metrics_tmp.log


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

        # metrics 값 추출
        SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
        PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
        LPIPS=$(grep -oP 'LPIPS\s*:\s*\K[0-9.e+-]+' metrics_tmp.log)
        MIOU=$(grep "mIoU" metrics_tmp.log | awk '{print $3}')

        python ../../update_sheet.py "$SHEET_NAME" "$SCENE" "$SSIM" "$PSNR" "$LPIPS" "$MIOU" "$TRAIN_TIME" "$RENDER_TIME" "$VRAM_MAX" "$GAUSSIAN_COUNT"


        # CSV 작성
        if [ ! -f "$CSV_FILE" ]; then
            echo "scene,SSIM,PSNR,LPIPS,MIOU" > "$CSV_FILE"
        fi
        echo "$SCENE,$SSIM,$PSNR,$LPIPS,$MIOU" >> "$CSV_FILE"

        echo "Metrics for $SCENE appended to $CSV_FILE"
        echo "Finished: $SCENE"
        echo
    fi
done

echo "All scenes processed."
