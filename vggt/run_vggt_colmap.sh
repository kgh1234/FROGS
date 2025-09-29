#!/bin/bash

# ./run_vggt_multi_scene.sh /path/to/root_folder

TOP_DIR=$1

if [ -z "$TOP_DIR" ]; then
  echo "Usage: $0 /path/to/folder"
  exit 1
fi

# 옵션 설정 (필요 시 수정)
USE_BA=false
MAX_REPROJ_ERROR=8.0
SHARED_CAMERA=false
CAMERA_TYPE="SIMPLE_PINHOLE"
VIS_THRESH=0.2
QUERY_FRAME_NUM=8
MAX_QUERY_PTS=4096
FINE_TRACKING=true
CONF_THRES_VALUE=1.0 # 압축 데이터가 너무 Sparse 해서 기본 5.0 -> 1.0 으로 변경함
SEED=42


for SCENE_DIR in "$TOP_DIR"/*/; do
  echo "Processing scene: $SCENE_DIR"


  python demo_colmap.py \
    --scene_dir "$SCENE_DIR" \
    --seed $SEED \
    $( [ "$USE_BA" = true ] && echo "--use_ba" ) \
    --max_reproj_error $MAX_REPROJ_ERROR \
    $( [ "$SHARED_CAMERA" = true ] && echo "--shared_camera" ) \
    --camera_type $CAMERA_TYPE \
    --vis_thresh $VIS_THRESH \
    --query_frame_num $QUERY_FRAME_NUM \
    --max_query_pts $MAX_QUERY_PTS \
    $( [ "$FINE_TRACKING" = true ] && echo "--fine_tracking" ) \
    --conf_thres_value $CONF_THRES_VALUE

  echo "Done with $SCENE_DIR"
  echo "---------------------------------------------"
done
