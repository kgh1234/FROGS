#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
# =============================================

SCENE_NAME="mipnerf"
ROOT="../../masked_datasets/$SCENE_NAME"
OUTPUT_ROOT="../../output_newloss_0.00001/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_$SCENE_NAME.csv"
SHEET_NAME="Ours_inoutloss_0.00001"

export CUDA_VISIBLE_DEVICES=0

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")
        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/mask"
        ORI_DIR="$SCENE_PATH/images_ori"
        OUT_DIR="$OUTPUT_ROOT/${SCENE}_original_masked_3dgs"

        echo "Evaluating metrics: $SCENE"
        python metrics_object.py -m "$OUT_DIR" -mask "$MASK_DIR" | tee metrics_tmp.log

        # metrics 값 추출
        SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
        PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
        LPIPS=$(grep "LPIPS" metrics_tmp.log | awk '{print $2}')

        python ../../update_sheet.py "$SHEET_NAME" "$SCENE" "$SSIM" "$PSNR" "$LPIPS" "$TRAIN_TIME" "$RENDER_TIME" "$VRAM_MAX"



    echo "Metrics for $SCENE appended to $CSV_FILE and Google Sheet"
    echo "Finished: $SCENE"

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
