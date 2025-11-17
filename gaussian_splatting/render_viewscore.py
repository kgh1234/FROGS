import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import itertools
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets"""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / (union + 1e-8)



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    visible_dir = os.path.join(model_path, name, f"ours_{iteration}", "visible_gaussians")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(visible_dir, exist_ok=True)
    
    all_visible_sets = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # --- 렌더 1회 호출 ---
        out_dict = render(view, gaussians, pipeline, background,
                          use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = out_dict["render"]
        vis_filter = out_dict.get("visibility_filter", None)

        # --- 디버그: 키 확인 ---
        if idx == 0:
            print(f"[DEBUG] out_dict keys: {list(out_dict.keys())}")

        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # --- 렌더 & GT 저장 ---
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

        # --- visibility_filter 기반 visible IDs 저장 ---
                # --- visibility IDs 저장 ---
        if vis_filter is not None:
            visible_ids = torch.nonzero(vis_filter, as_tuple=False).squeeze(-1).cpu().numpy().ravel().tolist()
        else:
            visible_ids = []

        vis_file = os.path.join(visible_dir, f"{idx:05d}.txt")
        with open(vis_file, "w") as f:
            f.write(" ".join(map(str, visible_ids)))

        all_visible_sets.append(set(visible_ids))



    n = len(all_visible_sets)
    if n > 1:
        view_jaccards = []
        for i in range(n):
            scores = []
            for j in range(n):
                if i == j:
                    continue
                scores.append(jaccard_similarity(all_visible_sets[i], all_visible_sets[j]))
            mean_j = sum(scores) / len(scores) if scores else 0
            view_jaccards.append(mean_j)

        # 콘솔 출력
        print("\n[INFO] Per-view Jaccard similarities:")
        for i, score in enumerate(view_jaccards):
            print(f"  View {i:05d}: {score:.4f}")

        # 전체 평균도 함께 출력
        overall_mean = sum(view_jaccards) / len(view_jaccards)
        print(f"\n[INFO] Mean Jaccard similarity ({name}): {overall_mean:.4f}")


    print(f"[INFO] Visibility sets saved to {visible_dir}")



def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--test_list", type=str, default="", help="파일명 목록(한 줄 하나). 여기에 포함된 train 카메라만 test로 렌더")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration,
                pipeline.extract(args), args.skip_train,
                args.skip_test, SPARSE_ADAM_AVAILABLE)
