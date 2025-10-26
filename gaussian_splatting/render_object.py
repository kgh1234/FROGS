#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, shutil, tempfile
from os import makedirs
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torchvision

from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams
from scene import Scene


def render_set_only_renders(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, out_name=None):
    tag = f"ours_{iteration}"
    render_root = os.path.join(model_path, name, tag, (out_name if out_name else "renders"))
    makedirs(render_root, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}/{tag}")):
        out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=False)
        img = out["render"]
        if train_test_exp:
            img = img[..., img.shape[-1] // 2:]
        torchvision.utils.save_image(img, os.path.join(render_root, f"{idx:05d}.png"))


def main():
    # ---- CLI wrapper ----
    cli = ArgumentParser("Render from custom camera JSON (novel views only, no GT required)")
    cli.add_argument("--model_path", "-m", required=True, help="Trained model folder (3DGS -m)")
    cli.add_argument("--camera_json", required=True, help="Path to your camera.json (NeRF/3DGS transforms-like)")
    cli.add_argument("--iteration", type=int, default=-1, help="Model iteration to load (e.g., 30000)")
    cli.add_argument("--images_ext", default=".jpg", help="Assume this ext if file_path has no ext (e.g., .jpg/.png)")
    cli.add_argument("--white_background", action="store_true", help="Force white background")
    cli.add_argument("--quiet", action="store_true")
    cli.add_argument("--out_name", default="custom_render", help="Subfolder name under test/ours_xxxxx/")
    # 품질/안정성용 추가 인자
    cli.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree used in training (default: 3)")
    cli.add_argument("--resolution", type=float, default=-1, help="-1 keeps ~1600px width cap; 1=full, 2=half, 4=quarter, ...")
    args_cli = cli.parse_args()

    # ---- check camera.json ----
    if not os.path.isfile(args_cli.camera_json):
        raise FileNotFoundError(f"camera_json not found: {args_cli.camera_json}")

    # ---- write camera.json -> tmp/transforms_test.json + empty transforms_train.json ----
    tmpdir = tempfile.mkdtemp(prefix="3dgs_custom_")
    tf_test  = os.path.join(tmpdir, "transforms_test.json")
    tf_train = os.path.join(tmpdir, "transforms_train.json")

    with open(args_cli.camera_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in ["fl_x", "fl_y", "cx", "cy", "w", "h", "frames"]:
        if k not in data:
            raise ValueError(f"camera_json missing required field: {k}")

    # test에는 camera.json 그대로
    with open(tf_test, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # train은 빈 frames로 생성(이미지 탐색 자체를 피함)
    empty_train = {
        "fl_x": data["fl_x"], "fl_y": data["fl_y"],
        "cx": data["cx"], "cy": data["cy"],
        "w": data["w"], "h": data["h"],
        "camera_angle_x": data.get("camera_angle_x", None),
        "camera_angle_y": data.get("camera_angle_y", None),
        "camera_model": data.get("camera_model", "OPENCV"),
        "frames": []
    }
    with open(tf_train, "w", encoding="utf-8") as f:
        json.dump(empty_train, f, indent=2)

    # ---- Build 3DGS parsers & defaults ----
    parser = ArgumentParser(add_help=False)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # parse default/empty, then set required fields manually
    args = parser.parse_args([])

    # required fields for Scene/loader
    args.source_path = tmpdir
    args.model_path  = args_cli.model_path
    args.images      = args_cli.images_ext   # used when file_path has no extension
    args.eval        = True                  # enable test split
    args.skip_train  = True                  # skip train (no GT)
    args.skip_test   = False                 # render test
    args.iteration   = args_cli.iteration
    args.quiet       = args_cli.quiet
    args.depths      = ""                    # must be string
    # defaults
    if not hasattr(args, "white_background"):
        args.white_background = False
    if args_cli.white_background:
        args.white_background = True
    if not hasattr(args, "train_test_exp"):
        args.train_test_exp = False
    # 메모리 안정성 옵션
    args.data_device = "cpu"                 # 카메라/이미지 텐서는 CPU (VRAM OOM 회피)
    # 해상도 파이프라인 반영
    args.resolution  = args_cli.resolution
    if not hasattr(args, "downsample") or args.downsample is None:
        args.downsample = 1.0
    # 🔴 SH 차수 기본값 강제 (None 방지)
    args.sh_degree = int(args_cli.sh_degree)

    print(f"[Info] model_path        : {args.model_path}")
    print(f"[Info] source_path (tmp) : {args.source_path}")
    print(f"[Info] images ext assume : {args.images}")
    print(f"[Info] iteration         : {args.iteration}")
    print(f"[Info] white background  : {getattr(args, 'white_background', False)}")

    # 확장자 힌트를 환경변수로 리더에게 전달 (.jpg/.png 등)
    os.environ["GS_IMAGES_EXT"] = args.images

    # ---- init & load ----
    safe_state(args.quiet)
    with torch.no_grad():
        # extract()는 내부 파서 기본값 적용하면서 필요한 필드만 살려서 전달
        extracted_model_args = model.extract(args)
        extracted_model_args.sh_degree = args.sh_degree  # 안전빵으로 한 번 더 보강

        g  = GaussianModel(extracted_model_args.sh_degree)
        sc = Scene(extracted_model_args, g, load_iteration=args.iteration, shuffle=False)

        bg_color   = [1,1,1] if extracted_model_args.white_background else [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # ---- render ONLY test views; save only renders ----
        extracted_pipeline_args = pipeline.extract(args)
        render_set_only_renders(
            model_path=extracted_model_args.model_path,
            name="test",
            iteration=sc.loaded_iter,
            views=sc.getTestCameras(),
            gaussians=g,
            pipeline=extracted_pipeline_args,
            background=background,
            train_test_exp=extracted_model_args.train_test_exp,
            out_name=args_cli.out_name
        )

    # cleanup
    try:
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"[Warn] temp dir not removed: {tmpdir} ({e})")


if __name__ == "__main__":
    main()
