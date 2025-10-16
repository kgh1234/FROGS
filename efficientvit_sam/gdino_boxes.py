#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GroundingDINO로 텍스트 프롬프트에 해당하는 바운딩박스를 추론하여 JSON으로 저장.
- 단일 이미지 또는 디렉터리 배치 처리 지원
- 출력 JSON 스키마:
  {
    "image": "<입력 이미지 경로>",
    "text": "<입력 텍스트>",
    "boxes_xyxy": [[x1,y1,x2,y2], ...],   # float, 원해상도 기준
    "logits": [float, ...],               # 각 박스 점수 (model logit)
    "phrases": ["table", ...]             # 매칭된 프레이즈
  }
"""

import os
import json
import argparse
from typing import List, Tuple
import numpy as np
from PIL import Image

# GroundingDINO 간단 추론 유틸
# (gdino_env에서 pip install -e . 한 상태여야 import 가능)
from groundingdino.util.inference import Model as GroundingDINOModel


def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]


def run_gdino_on_image(
    model: GroundingDINOModel,
    image_path: str,
    text: str,
    box_thr: float,
    text_thr: float,
    phrase_filter: str = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """단일 이미지에 대해 GroundingDINO 실행."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    result = model.predict_with_caption(
        image=img_np,
        caption=text,
        box_threshold=box_thr,
        text_threshold=text_thr,
    )

    if len(result) == 3:
        boxes, logits, phrases = result
    elif len(result) == 2:
        boxes, phrases = result
        logits = None
    else:
        raise RuntimeError(f"Unexpected GroundingDINO return format: {type(result)}")

    # phrase substring 필터링 (예: "table"만)
    if phrase_filter is not None:
        keep = [i for i, p in enumerate(phrases) if phrase_filter.lower() in p.lower()]
        if len(keep) == 0:
            return np.empty((0, 4), dtype=float), np.empty((0,), dtype=float), []
        boxes = boxes[keep]
        logits = logits[keep]
        phrases = [phrases[i] for i in keep]

    return boxes, logits, phrases


def save_json(
    out_path: str,
    image_path: str,
    text: str,
    boxes: np.ndarray,
    logits: np.ndarray,
    phrases: List[str],
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "image": image_path,
        "text": text,
        "boxes_xyxy": boxes.astype(float).tolist(),
        "logits": [float(x) for x in (logits.tolist() if hasattr(logits, "tolist") else logits)],
        "phrases": phrases,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description="GroundingDINO box exporter → JSON")
    p.add_argument("--image", type=str, help="단일 이미지 경로")
    p.add_argument("--dir", type=str, help="이미지 폴더 (배치 처리)")
    p.add_argument("--text", type=str, required=True, help='텍스트 프롬프트 (예: "table.")')
    p.add_argument("--config", type=str, required=True, help="GroundingDINO config .py")
    p.add_argument("--ckpt", type=str, required=True, help="GroundingDINO checkpoint .pth")
    p.add_argument("--out", type=str, required=True, help="출력 경로 (image 모드: json 파일, dir 모드: 출력 폴더)")
    p.add_argument("--box-thr", type=float, default=0.30, help="box threshold")
    p.add_argument("--text-thr", type=float, default=0.25, help="text threshold")
    p.add_argument("--phrase-filter", type=str, default=None, help='프레이즈 필터 (예: "table")')
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="장치")
    args = p.parse_args()

    if (args.image is None) == (args.dir is None):
        raise SystemExit("하나만 지정하세요: --image 또는 --dir")

    # 모델 로드
    model = GroundingDINOModel(
        model_config_path=args.config,
        model_checkpoint_path=args.ckpt,
        device=args.device,
    )

    if args.image:
        if not os.path.isfile(args.image):
            raise SystemExit(f"이미지 없음: {args.image}")
        boxes, logits, phrases = run_gdino_on_image(
            model, args.image, args.text, args.box_thr, args.text_thr, args.phrase_filter
        )
        if args.out.lower().endswith(".json"):
            out_json = args.out
        else:
            # 폴더가 들어오면 파일명 자동 생성
            os.makedirs(args.out, exist_ok=True)
            stem = os.path.splitext(os.path.basename(args.image))[0]
            out_json = os.path.join(args.out, f"{stem}.json")
        save_json(out_json, args.image, args.text, boxes, logits, phrases)
        print(f"✅ saved: {out_json} (boxes={len(phrases)})")

    else:
        # dir 배치
        if not os.path.isdir(args.out):
            os.makedirs(args.out, exist_ok=True)
        images = [f for f in os.listdir(args.dir) if is_image_file(f)]
        images.sort()
        if not images:
            raise SystemExit(f"이미지 폴더 비어있음: {args.dir}")

        cnt_total, cnt_nonempty = 0, 0
        for name in images:
            img_path = os.path.join(args.dir, name)
            boxes, logits, phrases = run_gdino_on_image(
                model, img_path, args.text, args.box_thr, args.text_thr, args.phrase_filter
            )
            stem = os.path.splitext(name)[0]
            out_json = os.path.join(args.out, f"{stem}.json")
            save_json(out_json, img_path, args.text, boxes, logits, phrases)
            cnt_total += 1
            if len(phrases) > 0:
                cnt_nonempty += 1
            print(f"✅ {name} → {os.path.relpath(out_json)} (boxes={len(phrases)})")

        print(f"\n�� done: {cnt_nonempty}/{cnt_total} images had at least one box.")


if __name__ == "__main__":
    main()
