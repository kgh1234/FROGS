import os, glob, cv2, torch
from argparse import ArgumentParser
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state

def latest_iter(model_path):
    it_dirs = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
    cand = []
    for d in it_dirs:
        try:
            cand.append(int(os.path.basename(d).split("_")[-1]))
        except:
            pass
    return max(cand) if cand else -1

def save_img(tensor_chw, out_path):
    img = (torch.clamp(tensor_chw, 0.0, 1.0) * 255).byte().permute(1,2,0).cpu().numpy()  # HWC, RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--test_list", type=str, default="", help="한 줄에 하나씩 이미지 파일명(.JPG 등) 목록")
    parser.add_argument("--iteration", type=int, default=-1, help="로딩할 체크포인트. -1이면 최신 자동 탐색")
    parser.add_argument("--skip_train", action="store_true", help="train 렌더 건너뛰기")
    parser.add_argument("--tag", type=str, default="ours", help="출력 폴더 태그 (기본 ours)")

    args = parser.parse_args()
    safe_state(True)

    dataset = lp.extract(args)
    pipe     = pp.extract(args)

    # 로드할 이터레이션 결정
    it = args.iteration if args.iteration >= 0 else latest_iter(dataset.model_path)
    print(f"[INFO] Using iteration = {it}")

    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="adam")
    scene = Scene(dataset, gaussians, load_iteration=it, shuffle=False)

    # 배경
    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ===== 1) TRAIN 렌더 =====
    if not args.skip_train:
        train_out = os.path.join(dataset.model_path, "train", f"{args.tag}_{it}")
        cams = scene.getTrainCameras()
        print(f"[INFO] Rendering TRAIN: {len(cams)} views -> {train_out}")
        for cam in tqdm(cams, desc="Train"):
            pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
            img = pkg["render"]
            name = os.path.splitext(getattr(cam, "image_name", "view"))[0] + ".png"
            save_img(img, os.path.join(train_out, name))

    # ===== 2) TEST(부분집합) 렌더 =====
    test_sel = set()
    if args.test_list and os.path.isfile(args.test_list):
        with open(args.test_list, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    test_sel.add(line)
    else:
        print("[WARN] --test_list 비어있거나 파일이 없습니다. test는 건너뜀.")
        return

    # train 카메라 중 파일명이 test_list에 포함된 것만 선별
    train_cams = scene.getTrainCameras()
    def match(cam):
        nm = getattr(cam, "image_name", "")
        base = nm.split('/')[-1]
        return (nm in test_sel) or (base in test_sel)
    test_cams = [c for c in train_cams if match(c)]

    test_out = os.path.join(dataset.model_path, "test", f"{args.tag}_{it}")
    print(f"[INFO] Rendering TEST(subset): {len(test_cams)} views -> {test_out}")
    for cam in tqdm(test_cams, desc="Test"):
        pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
        img = pkg["render"]
        name = os.path.splitext(getattr(cam, "image_name", "view"))[0] + ".png"
        save_img(img, os.path.join(test_out, name))

if __name__ == "__main__":
    main()
