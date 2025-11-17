#!/bin/bash
# =============================================
# 🔄 자동 3DGS 파이프라인 (Train → JSON → Render)
# ROOT 폴더 내 모든 scene 순회 실행
# =============================================

# ===== 기본 설정 =====
ROOT="../../masked_datasets/lerf_mask"
OUTPUT_ROOT="../../output_speedy/lerf_mask"
ORI_ROOT="../../output_original/lerf_mask"
GT_OUTPUT_ROOT="../../output_mipsplatting_ori"
GT_ROOT="../../output_mipsplatting"
PLY_ITER=30000
NUM_VIEWS=60
IMG_W=986
IMG_H=728
KERNEL_SIZE=1.0
source /opt/conda/etc/profile.d/conda.sh

# ===== SCENE 반복 =====
for SCENE_PATH in "${ROOT}"/*; do
    if [ -d "$SCENE_PATH" ]; then
       SCENE_PATH=
        SCENE_NAME=$(basename "$SCENE_PATH")

        OBJECT_NUMBER=${SCENE_NAME##*_}
        GT_SCENE_NAME=${SCENE_NAME%%_*}
        GT_PATH="${GT_ROOT}/${GT_SCENE_NAME}"
        GT_OUTPUT_PATH="${GT_OUTPUT_ROOT}/${GT_SCENE_NAME}"
    
        OUTPUT_PATH="${OUTPUT_ROOT}/${SCENE_NAME}"
        
        LATEST_DATE_DIR=$(ls -d "${OUTPUT_PATH}"/*/ 2>/dev/null | sort | tail -n 1)


        GT_PLY_PATH="${OUTPUT_PATH}/point_cloud/iteration_${PLY_ITER}/point_cloud.ply"



        PLY_PATH=$(ls -td "${LATEST_DATE_DIR}"/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | head -n 1)
        JSON_PATH="${OUTPUT_PATH}/transforms_test.json"


        ITER_DIR_NAME=$(basename "$(dirname "${PLY_PATH}")")
        ITER_NUM_OURS=$(echo "${ITER_DIR_NAME}" | grep -oE '[0-9]+')

        # conda activate mip-splat-cu121

        # echo "====================================="
        # echo "▶ [1] Training: ${SCENE_NAME}"
        # echo "====================================="
        # python ../mip-splatting/train.py \
        #     -s "${SCENE_PATH}" \
        #     -m "${OUTPUT_PATH}"

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

        echo OBJECT_NUMBER: ${OBJECT_NUMBER}

        echo "====================================="
        echo "▶ [3] Rendering: ${SCENE_NAME}"
        echo "====================================="


        Pseudo GT rendering
        python render_object.py \
            -m "${GT_OUTPUT_PATH}" \
            --camera_json "${JSON_PATH}" \
            --iteration ${PLY_ITER} \
            --images_ext .jpg \
            --out_name hemisphere_render \
            --ply_path "${GT_PLY_PATH}" \
            --object_number ${OBJECT_NUMBER}

        # # ours Rendering
        # python render_object.py \
        #     -m "${LATEST_DATE_DIR}" \
        #     --camera_json "${JSON_PATH}" \
        #     --iteration ${ITER_NUM_OURS} \
        #     --images_ext .jpg \
        #     --out_name hemisphere_render

        ORIGINAL_OUTPUT_PATH="${ORI_ROOT}/${GT_SCENE_NAME}"
        PLY_ITER=30000
        # 3DGS Rendering
        python render_object.py \
            -m "${ORIGINAL_OUTPUT_PATH}" \
            --camera_json "${JSON_PATH}" \
            --iteration ${PLY_ITER} \
            --images_ext .jpg \
            --out_name hemisphere_render --object_number ${OBJECT_NUMBER}


        echo "====================================="
        echo "완료: ${SCENE_NAME} 파이프라인 종료"
        echo "결과 위치: ${OUTPUT_PATH}"
        echo "====================================="
        echo ""
    fi
done

echo "🎉 모든 SCENE 실행 완료!"
