#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# 3회 반복 실행 후 평균값 기록 (+ Gaussian 수 포함)
# =============================================

SCENE_NAME="mipnerf"
ROOT="../../masked_datasets/$SCENE_NAME"
OUTPUT_ROOT="../../output_ours/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_${SCENE_NAME}.csv"
SHEET_NAME="Re_Ours"

export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_ROOT"

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE_PATH="../../masked_datasets/mipnerf/garden"
        SCENE=$(basename "$SCENE_PATH")
        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/mask"

        echo "====================================="
        echo "Processing scene: $SCENE"
        echo "====================================="

        if [ "$SCENE" == "figurines_15" ] ; then
            echo "Skipping scene: $SCENE"
            continue
        fi

        # 누적값 초기화
        TOTAL_SSIM=0
        TOTAL_PSNR=0
        TOTAL_LPIPS=0
        TOTAL_TRAIN_TIME=0
        TOTAL_RENDER_TIME=0
        TOTAL_VRAM=0
        TOTAL_GAUSSIAN=0

        for i in {1..3}; do
            echo "[Run $i/3] $SCENE 시작"
            OUT_DIR="$OUTPUT_ROOT/${SCENE}_run${i}"
            LOGFILE="vram_${SCENE}_run${i}.log"

            TRAIN_START=$(date +%s)
            nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -l 2 > "$LOGFILE" &
            VRAM_PID=$!

            python train_all.py -s "$SCENE_PATH" -m "$OUT_DIR" --mask_dir "$MASK_DIR" --eval 

            TRAIN_END=$(date +%s)
            TRAIN_TIME=$((TRAIN_END - TRAIN_START))
            kill $VRAM_PID 2>/dev/null
            VRAM_MAX=$(awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "$LOGFILE")
            rm -f "$LOGFILE"

            echo "[Run $i/3] Rendering..."
            RENDER_START=$(date +%s)
            python render_with_mask.py -m "$OUT_DIR" | tee render_tmp.log
            RENDER_END=$(date +%s)
            RENDER_TIME=$((RENDER_END - RENDER_START))

            echo "[Run $i/3] Evaluating metrics..."
            python metrics_object_mIoU.py -m "$OUT_DIR" --mask_dir "$MASK_DIR" | tee metrics_tmp.log

            # Gaussian 개수 확인
            POINT_CLOUD_DIR="$OUT_DIR/point_cloud"
            GAUSSIAN_COUNT=0
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

            SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
            PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
            LPIPS=$(grep "LPIPS" metrics_tmp.log | awk '{print $2}')
            MIOU=$(grep "mIoU" metrics_tmp.log | awk '{print $3}')

            # 누적 합산
            TOTAL_SSIM=$(echo "$TOTAL_SSIM + $SSIM" | bc)
            TOTAL_PSNR=$(echo "$TOTAL_PSNR + $PSNR" | bc)
            TOTAL_LPIPS=$(echo "$TOTAL_LPIPS + $LPIPS" | bc)
            TOTAL_MIOU=$(echo "$TOTAL_MIOU + $MIOU" | bc)
            TOTAL_TRAIN_TIME=$((TOTAL_TRAIN_TIME + TRAIN_TIME))
            TOTAL_RENDER_TIME=$((TOTAL_RENDER_TIME + RENDER_TIME))
            TOTAL_VRAM=$((TOTAL_VRAM + VRAM_MAX))
            TOTAL_GAUSSIAN=$((TOTAL_GAUSSIAN + GAUSSIAN_COUNT))

            # CSV 개별 회차 기록
            if [ ! -f "$CSV_FILE" ]; then
                echo "scene,run,SSIM,PSNR,LPIPS,MIOU,TrainTime(s),RenderTime(s),MaxVRAM(MB),GaussianCount" > "$CSV_FILE"
            fi
            echo "$SCENE,run${i},$SSIM,$PSNR,$LPIPS,$MIOU,$TRAIN_TIME,$RENDER_TIME,$VRAM_MAX,$GAUSSIAN_COUNT" >> "$CSV_FILE"

            echo "[Run $i/3] 완료. GPU 쿨다운 중..."
            sleep 10
            nvidia-smi -q -d TEMPERATURE | grep "GPU Current Temp"

        done

        # 평균 계산
        AVG_SSIM=$(echo "scale=3; $TOTAL_SSIM / 3" | bc)
        AVG_PSNR=$(echo "scale=3; $TOTAL_PSNR / 3" | bc)
        AVG_LPIPS=$(echo "scale=3; $TOTAL_LPIPS / 3" | bc)
        AVG_MIOU=$(echo "scale=3; $TOTAL_MIOU / 3" | bc)
        AVG_TRAIN_TIME=$((TOTAL_TRAIN_TIME / 3))
        AVG_RENDER_TIME=$((TOTAL_RENDER_TIME / 3))
        AVG_VRAM=$((TOTAL_VRAM / 3))
        AVG_GAUSSIAN=$((TOTAL_GAUSSIAN / 3))

        # Google Sheet 업데이트 (평균만 기록)
        python ../../update_sheet.py "$SHEET_NAME" "$SCENE" "$AVG_SSIM" "$AVG_PSNR" "$AVG_LPIPS" "$AVG_MIOU" "$AVG_TRAIN_TIME" "$AVG_RENDER_TIME" "$AVG_VRAM" "$AVG_GAUSSIAN"

        echo "Averaged Metrics for $SCENE → Google Sheet updated."
        echo "====================================="
    fi
done

echo "All scenes processed."

SCENE_NAME="mipnerf"
ROOT="../../masked_datasets/$SCENE_NAME"
OUTPUT_ROOT="../../output_ours/$SCENE_NAME"
CSV_FILE="$OUTPUT_ROOT/metrics_summary_${SCENE_NAME}.csv"
SHEET_NAME="Re_Ours"

export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_ROOT"

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")
        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/mask"

        echo "====================================="
        echo "Processing scene: $SCENE"
        echo "====================================="

        # if [ "$SCENE" == "bicycle" ] ; then
        #     echo "Skipping scene: $SCENE"
        #     continue
        # fi

        # 누적값 초기화
        TOTAL_SSIM=0
        TOTAL_PSNR=0
        TOTAL_LPIPS=0
        TOTAL_TRAIN_TIME=0
        TOTAL_RENDER_TIME=0
        TOTAL_VRAM=0
        TOTAL_GAUSSIAN=0

        for i in {1..3}; do
            echo "[Run $i/3] $SCENE 시작"
            OUT_DIR="$OUTPUT_ROOT/${SCENE}_run${i}"
            LOGFILE="vram_${SCENE}_run${i}.log"

            TRAIN_START=$(date +%s)
            nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -l 2 > "$LOGFILE" &
            VRAM_PID=$!

            python train_all.py -s "$SCENE_PATH" -m "$OUT_DIR" --mask_dir "$MASK_DIR" --eval 

            TRAIN_END=$(date +%s)
            TRAIN_TIME=$((TRAIN_END - TRAIN_START))
            kill $VRAM_PID 2>/dev/null
            VRAM_MAX=$(awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "$LOGFILE")
            rm -f "$LOGFILE"

            echo "[Run $i/3] Rendering..."
            RENDER_START=$(date +%s)
            python render_with_mask.py -m "$OUT_DIR" | tee render_tmp.log
            RENDER_END=$(date +%s)
            RENDER_TIME=$((RENDER_END - RENDER_START))

            echo "[Run $i/3] Evaluating metrics..."
            python metrics_object_mIoU.py -m "$OUT_DIR" --mask_dir "$MASK_DIR" | tee metrics_tmp.log

            # Gaussian 개수 확인
            POINT_CLOUD_DIR="$OUT_DIR/point_cloud"
            GAUSSIAN_COUNT=0
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

            SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
            PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
            LPIPS=$(grep "LPIPS" metrics_tmp.log | awk '{print $2}')
            MIOU=$(grep "mIoU" metrics_tmp.log | awk '{print $3}')

            # 누적 합산
            TOTAL_SSIM=$(echo "$TOTAL_SSIM + $SSIM" | bc)
            TOTAL_PSNR=$(echo "$TOTAL_PSNR + $PSNR" | bc)
            TOTAL_LPIPS=$(echo "$TOTAL_LPIPS + $LPIPS" | bc)
            TOTAL_MIOU=$(echo "$TOTAL_MIOU + $MIOU" | bc)
            TOTAL_TRAIN_TIME=$((TOTAL_TRAIN_TIME + TRAIN_TIME))
            TOTAL_RENDER_TIME=$((TOTAL_RENDER_TIME + RENDER_TIME))
            TOTAL_VRAM=$((TOTAL_VRAM + VRAM_MAX))
            TOTAL_GAUSSIAN=$((TOTAL_GAUSSIAN + GAUSSIAN_COUNT))

            # CSV 개별 회차 기록
            if [ ! -f "$CSV_FILE" ]; then
                echo "scene,run,SSIM,PSNR,LPIPS,MIOU,TrainTime(s),RenderTime(s),MaxVRAM(MB),GaussianCount" > "$CSV_FILE"
            fi
            echo "$SCENE,run${i},$SSIM,$PSNR,$LPIPS,$MIOU,$TRAIN_TIME,$RENDER_TIME,$VRAM_MAX,$GAUSSIAN_COUNT" >> "$CSV_FILE"

            echo "[Run $i/3] 완료. GPU 쿨다운 중..."
            sleep 10
            nvidia-smi -q -d TEMPERATURE | grep "GPU Current Temp"

        done

        # 평균 계산
        AVG_SSIM=$(echo "scale=3; $TOTAL_SSIM / 3" | bc)
        AVG_PSNR=$(echo "scale=3; $TOTAL_PSNR / 3" | bc)
        AVG_LPIPS=$(echo "scale=3; $TOTAL_LPIPS / 3" | bc)
        AVG_MIOU=$(echo "scale=3; $TOTAL_MIOU / 3" | bc)
        AVG_TRAIN_TIME=$((TOTAL_TRAIN_TIME / 3))
        AVG_RENDER_TIME=$((TOTAL_RENDER_TIME / 3))
        AVG_VRAM=$((TOTAL_VRAM / 3))
        AVG_GAUSSIAN=$((TOTAL_GAUSSIAN / 3))

        # Google Sheet 업데이트 (평균만 기록)
        python ../../update_sheet.py "$SHEET_NAME" "$SCENE" "$AVG_SSIM" "$AVG_PSNR" "$AVG_LPIPS" "$AVG_MIOU" "$AVG_TRAIN_TIME" "$AVG_RENDER_TIME" "$AVG_VRAM" "$AVG_GAUSSIAN"

        echo "Averaged Metrics for $SCENE → Google Sheet updated."
        echo "====================================="
    fi
done

echo "All scenes processed."