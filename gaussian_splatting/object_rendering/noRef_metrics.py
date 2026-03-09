#!/usr/bin/env python3
"""
이미지 품질 평가 스크립트
─────────────────────────────────────────────────────
지표:
  ① CLIPScore   - CLIP ViT-B/32, 렌더 ↔ 레퍼런스 코사인 유사도  (Pairwise)
  ② DINOScore   - DINOv2 ViT-B/14, 렌더 ↔ 레퍼런스 코사인 유사도 (Pairwise)
  ③ CLIP-IQA+   - torchmetrics CLIPImageQualityAssessment        (No-Ref)
  ④ MUSIQ       - torchmetrics MultiScaleImageQualityAssessor    (No-Ref)

출력:
  --out_json  : scene 별 상세 결과 JSON (per-image 점수 포함)
  --out_csv   : 전체 취합 CSV에 한 row 추가 (pipeline 의 [5]에서 집계)
─────────────────────────────────────────────────────
"""

import os
import csv
import glob
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


# ═══════════════════════════════════════════════════
# 모델 팩토리
# ═══════════════════════════════════════════════════

def get_clip_model(device):
    try:
        import clip
    except ImportError:
        raise ImportError("pip install git+https://github.com/openai/CLIP.git")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def get_dino_model(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def get_clip_iqa_model(device):
    """
    torchmetrics >= 1.0 의 CLIPImageQualityAssessment (CLIP-IQA+).
    입력: uint8 텐서 (B, 3, H, W)  값 범위 [0, 255]
    출력: float score (높을수록 품질 좋음, 범위 0~1)
    """
    try:
        from torchmetrics.multimodal import CLIPImageQualityAssessment
    except ImportError:
        raise ImportError(
            "CLIP-IQA 설치 필요:\n"
            "  pip install torchmetrics[multimodal]\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )
    metric = CLIPImageQualityAssessment(model_name_or_path="clip_iqa+").to(device)
    metric.eval()

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),                                       # [0,1] float
        transforms.Lambda(lambda t: (t * 255).to(torch.uint8)),     # uint8
    ])
    return metric, preprocess


def get_musiq_model(device):
    """
    torchmetrics >= 1.0 의 MultiScaleImageQualityAssessor (MUSIQ).
    가중치: musiq-paq2piq  (점수 범위 0~100)
    입력: uint8 텐서 (B, 3, H, W)  값 범위 [0, 255]
    """
    try:
        from torchmetrics.image.musiq import MultiScaleImageQualityAssessor
    except ImportError:
        raise ImportError(
            "MUSIQ 설치 필요:\n"
            "  pip install torchmetrics[image]"
        )
    metric = MultiScaleImageQualityAssessor(model_name="musiq-paq2piq").to(device)
    metric.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 255).to(torch.uint8)),
    ])
    return metric, preprocess


# ═══════════════════════════════════════════════════
# 이미지 유틸
# ═══════════════════════════════════════════════════

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def collect_images(directory):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


def load_pil(path):
    return Image.open(path).convert("RGB")


def pair_images(rendered, refs):
    """렌더 수 기준으로 레퍼런스를 cycle 매칭"""
    n = len(rendered)
    if not refs:
        return rendered, []
    return rendered, [refs[i % len(refs)] for i in range(n)]


# ═══════════════════════════════════════════════════
# 점수 계산
# ═══════════════════════════════════════════════════

@torch.no_grad()
def compute_clip_scores(model, preprocess, device, rendered_paths, ref_paths):
    scores = []
    for rp, gp in zip(rendered_paths, ref_paths):
        ir = preprocess(load_pil(rp)).unsqueeze(0).to(device)
        ig = preprocess(load_pil(gp)).unsqueeze(0).to(device)
        fr = F.normalize(model.encode_image(ir), dim=-1)
        fg = F.normalize(model.encode_image(ig), dim=-1)
        scores.append(float((fr * fg).sum().item()))
    return scores


@torch.no_grad()
def compute_dino_scores(model, preprocess, device, rendered_paths, ref_paths):
    scores = []
    for rp, gp in zip(rendered_paths, ref_paths):
        ir = preprocess(load_pil(rp)).unsqueeze(0).to(device)
        ig = preprocess(load_pil(gp)).unsqueeze(0).to(device)
        fr = F.normalize(model(ir), dim=-1)
        fg = F.normalize(model(ig), dim=-1)
        scores.append(float((fr * fg).sum().item()))
    return scores


@torch.no_grad()
def compute_clip_iqa_scores(metric, preprocess, device, image_paths):
    """No-Reference: 렌더 이미지에만 적용"""
    scores = []
    for p in image_paths:
        img = preprocess(load_pil(p)).unsqueeze(0).to(device)  # (1,3,H,W) uint8
        out = metric(img)
        # torchmetrics 버전에 따라 dict 또는 tensor 반환
        val = list(out.values())[0] if isinstance(out, dict) else out
        scores.append(float(val.mean().item()))
    return scores


@torch.no_grad()
def compute_musiq_scores(metric, preprocess, device, image_paths):
    """No-Reference: 렌더 이미지에만 적용"""
    scores = []
    for p in image_paths:
        img = preprocess(load_pil(p)).unsqueeze(0).to(device)  # (1,3,H,W) uint8
        out = metric(img)
        scores.append(float(out.mean().item()))
    return scores


# ═══════════════════════════════════════════════════
# CSV 헬퍼
# ═══════════════════════════════════════════════════

CSV_FIELDS = [
    "scene",
    "num_rendered",
    "num_refs",
    "clip_score_mean",
    "dino_score_mean",
    "clip_iqa_mean",
    "musiq_mean",
]


def append_csv_row(csv_path, row: dict):
    """CSV 파일에 한 row 추가. 파일 없으면 헤더 포함 신규 생성."""
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k, ""))
                         for k in CSV_FIELDS})


# ═══════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="렌더 이미지 품질 평가 (CLIPScore / DINOScore / CLIP-IQA / MUSIQ)"
    )
    parser.add_argument("--render_dir",    required=True,  help="렌더링 결과 폴더")
    parser.add_argument("--ref_dir",       required=True,  help="레퍼런스 이미지 폴더")
    parser.add_argument("--out_json",      required=True,  help="scene JSON 결과 경로")
    parser.add_argument("--out_csv",       required=True,  help="전체 취합 CSV 경로")
    parser.add_argument("--scene_name",    default="",     help="scene 이름 (로그 및 CSV용)")
    # 지표별 skip
    parser.add_argument("--skip_clip",     action="store_true")
    parser.add_argument("--skip_dino",     action="store_true")
    parser.add_argument("--skip_clip_iqa", action="store_true")
    parser.add_argument("--skip_musiq",    action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device      : {device}")
    print(f"[eval] scene       : {args.scene_name}")
    print(f"[eval] render_dir  : {args.render_dir}")
    print(f"[eval] ref_dir     : {args.ref_dir}")

    rendered = collect_images(args.render_dir)
    refs     = collect_images(args.ref_dir)

    if not rendered:
        print(f"[eval] ERROR: 렌더링 이미지 없음 → {args.render_dir}")
        return

    rendered, matched_refs = pair_images(rendered, refs)
    has_refs = bool(matched_refs)

    print(f"[eval] rendered    : {len(rendered)} 장")
    print(f"[eval] refs        : {len(refs)} 장  (매칭: {len(matched_refs)} 장)")

    result = {
        "scene"       : args.scene_name,
        "render_dir"  : args.render_dir,
        "ref_dir"     : args.ref_dir,
        "num_rendered": len(rendered),
        "num_refs"    : len(refs),
    }

    # ── ① CLIPScore (Pairwise) ────────────────────────
    if not args.skip_clip:
        if not has_refs:
            print("[eval] ① CLIP SKIP : 레퍼런스 없음")
        else:
            print("[eval] ① CLIPScore 로딩...")
            m, p = get_clip_model(device)
            s = compute_clip_scores(m, p, device, rendered, matched_refs)
            result["clip_score_mean"]      = round(float(np.mean(s)), 6)
            result["clip_score_per_image"] = s
            print(f"[eval]   CLIPScore  = {result['clip_score_mean']:.4f}")
            del m; torch.cuda.empty_cache()

    # ── ② DINOScore (Pairwise) ───────────────────────
    if not args.skip_dino:
        if not has_refs:
            print("[eval] ② DINO SKIP : 레퍼런스 없음")
        else:
            print("[eval] ② DINOScore 로딩...")
            m, p = get_dino_model(device)
            s = compute_dino_scores(m, p, device, rendered, matched_refs)
            result["dino_score_mean"]      = round(float(np.mean(s)), 6)
            result["dino_score_per_image"] = s
            print(f"[eval]   DINOScore  = {result['dino_score_mean']:.4f}")
            del m; torch.cuda.empty_cache()

    # ── ③ CLIP-IQA+ (No-Reference) ──────────────────
    if not args.skip_clip_iqa:
        print("[eval] ③ CLIP-IQA+ 로딩...")
        try:
            m, p = get_clip_iqa_model(device)
            s = compute_clip_iqa_scores(m, p, device, rendered)
            result["clip_iqa_mean"]      = round(float(np.mean(s)), 6)
            result["clip_iqa_per_image"] = s
            print(f"[eval]   CLIP-IQA+  = {result['clip_iqa_mean']:.4f}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[eval]   CLIP-IQA+ 실패: {e}")

    # ── ④ MUSIQ (No-Reference) ───────────────────────
    if not args.skip_musiq:
        print("[eval] ④ MUSIQ 로딩...")
        try:
            m, p = get_musiq_model(device)
            s = compute_musiq_scores(m, p, device, rendered)
            result["musiq_mean"]      = round(float(np.mean(s)), 6)
            result["musiq_per_image"] = s
            print(f"[eval]   MUSIQ      = {result['musiq_mean']:.4f}  (0~100 scale)")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[eval]   MUSIQ 실패: {e}")

    # ── JSON 저장 ────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[eval] JSON 저장   : {args.out_json}")

    # ── CSV 행 추가 ──────────────────────────────────
    append_csv_row(args.out_csv, {
        "scene"          : result["scene"],
        "num_rendered"   : result["num_rendered"],
        "num_refs"       : result["num_refs"],
        "clip_score_mean": result.get("clip_score_mean"),
        "dino_score_mean": result.get("dino_score_mean"),
        "clip_iqa_mean"  : result.get("clip_iqa_mean"),
        "musiq_mean"     : result.get("musiq_mean"),
    })
    print(f"[eval] CSV 추가    : {args.out_csv}")


if __name__ == "__main__":
    main()