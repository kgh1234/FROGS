#!/bin/bash
# =============================================
# 🔄 자동 3DGS 파이프라인 (Train → JSON → Render)
# ROOT 폴더 내 모든 scene 순회 실행
# =============================================

# ===== 기본 설정 =====
ROOT="../../original_datasets/lerf_mask"
OUTPUT_ROOT="../../output_mipsplatting_ori"
PLY_ITER=30000
NUM_VIEWS=60
IMG_W=986
IMG_H=728
KERNEL_SIZE=1.0

source /opt/conda/etc/profile.d/conda.sh

# ===== SCENE 반복 =====
for SCENE_PATH in "${ROOT}"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE_NAME=$(basename "$SCENE_PATH")
        OUTPUT_PATH="${OUTPUT_ROOT}/${SCENE_NAME}"
        PLY_PATH="${OUTPUT_PATH}/point_cloud/iteration_${PLY_ITER}/point_cloud.ply"
        JSON_PATH="${OUTPUT_PATH}/transforms_test.json"

        conda activate mip-splat-cu121

        echo "====================================="
        echo "▶ [1] Training: ${SCENE_NAME}"
        echo "====================================="
        python ../mip-splatting/train.py \
            -s "${SCENE_PATH}" \
            -m "${OUTPUT_PATH}"

        echo "====================================="
        echo "▶ [2] Generating JSON for ${SCENE_NAME}"
        echo "====================================="
        python object_rendering/nerf_dir_camera.py \
            --ply "${PLY_PATH}" \
            --out "${JSON_PATH}" \
            --images_dir "${SCENE_PATH}/images" \
            --prefix frame_ --ext jpg --pad 5 --start 1 \
            --num_views ${NUM_VIEWS} --radius 2 \
            --img_w ${IMG_W} --img_h ${IMG_H} \
            --nerf_negz --absolute --store_no_ext --roll180 \
            --elev_start_deg -80 --elev_end_deg 0 --elev_steps 10

        conda activate 3dgs_gahyeon

        echo "====================================="
        echo "▶ [3] Rendering: ${SCENE_NAME}"
        echo "====================================="
        python render_object.py \
            -m "${OUTPUT_PATH}" \
            --camera_json "${JSON_PATH}" \
            --iteration ${PLY_ITER} \
            --images_ext .jpg \
            --out_name hemisphere_render

        echo "====================================="
        echo "완료: ${SCENE_NAME} 파이프라인 종료"
        echo "결과 위치: ${OUTPUT_PATH}"
        echo "====================================="
        echo ""
    fi
done

echo "🎉 모든 SCENE 실행 완료!"
