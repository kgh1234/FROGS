import os
import sys
import glob
import uuid
import torch
import numpy as np
import cv2
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

# ===== Optional Writers / Backends =====
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False



class AdaptiveStopper:
    def __init__(self, patience=1000, min_delta=1e-5):
        """
        patience : 최근 몇 step 동안의 변화를 추적할지
        min_delta: 평균 개선폭이 이 값보다 작으면 stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.history = []

    def update(self, loss_val):
        self.history.append(loss_val)
        if len(self.history) < self.patience:
            return False  # 아직 모니터링 구간 부족

        recent = np.array(self.history[-self.patience:])
        improvement = recent[:-1] - recent[1:]
        mean_improve = np.mean(improvement)

        if mean_improve < self.min_delta:
            print(f"[AdaptiveStop] Loss improvement too small ({mean_improve:.6e}) → stopping training.")
            return True
        return False


# =========================
# Mask Utilities
# =========================
def _stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]

def _find_mask_path(mask_dir: str, image_name_or_path: str):
    stem = _stem(image_name_or_path)
    for ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
        cand = os.path.join(mask_dir, f"{stem}.{ext}")
        if os.path.isfile(cand):
            return cand
    # fallback for “mask” token
    for ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
        cands = glob.glob(os.path.join(mask_dir, f"{stem}*mask*.{ext}"))
        if cands:
            return cands[0]
    return None

def _load_binary_mask(mask_path: str, H: int, W: int, binary_threshold=128, invert=False, device="cuda"):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if invert:
        m = 255 - m
    m = (m >= binary_threshold).astype(np.float32)
    return torch.from_numpy(m).to(device)  # (H, W)


# =========================
# Main Training Loop
# =========================
def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             mask_dir=None, mask_binary_threshold=128, mask_invert=False):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("SparseAdam optimizer unavailable — install correct rasterizer via `pip install [3dgs_accel]`.")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()

        # Network GUI (viewer)
        while network_gui.conn is not None:
            try:
                custom_cam, do_training, *_ = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background)["render"]
                    network_gui.send(
                        memoryview((torch.clamp(net_image, 0, 1.0) * 255).byte()
                                   .permute(1, 2, 0).contiguous().cpu().numpy()),
                        dataset.source_path
                    )
                if do_training:
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        # ===== Masked loss =====
        use_mask = mask_dir is not None and len(mask_dir) > 0
        if use_mask:
            mask_path = _find_mask_path(mask_dir, viewpoint_cam.image_name)
            if mask_path and os.path.exists(mask_path):
                mask = _load_binary_mask(mask_path, image.shape[1], image.shape[2],
                                         binary_threshold=mask_binary_threshold,
                                         invert=mask_invert).unsqueeze(0)
                mask = mask.expand_as(gt_image)
            else:
                mask = torch.ones_like(gt_image)
            diff = torch.abs(image - gt_image) * mask
            Ll1 = diff.sum() / (mask.sum() * image.shape[0] + 1e-8)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE \
                         else ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE \
                         else ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        stopper = stopper if 'stopper' in locals() else AdaptiveStopper(patience=10000, min_delta=1e-10)
        if stopper.update(loss.item()):
            print(f"Early stop triggered at iteration {iteration}")
            print(f"\n[ITER {iteration}] Saving Gaussians...")
            scene.save(iteration)
            break

        # Depth regularization
        Ll1depth = 0.0
        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "depth_reliable", False):
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth

        # Backward + step
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.6f}", "Depth": f"{ema_Ll1depth_for_log:.6f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background), dataset.train_test_exp,
                            mask_dir=mask_dir,
                            mask_binary_threshold=mask_binary_threshold,
                            mask_invert=mask_invert)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians...")
                scene.save(iteration)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    if not getattr(args, "model_path", None):
        unique_str = str(uuid.uuid4())[:10]
        args.model_path = os.path.join("./output/", unique_str)

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: no logging")
    return None


# =========================
# Validation
# =========================
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs,
                    train_test_exp, mask_dir=None, mask_binary_threshold=128, mask_invert=False):

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', float(Ll1), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', float(loss), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    validation_configs = [
        {'name': 'test', 'cameras': scene.getTestCameras()},
        {'name': 'train', 'cameras': [scene.getTrainCameras()[i] for i in range(5, min(30, len(scene.getTrainCameras())), 5)]}
    ]

    for config in validation_configs:
        if not config['cameras']:
            continue
        l1_test, psnr_test = 0.0, 0.0
        for idx, viewpoint in enumerate(config['cameras']):
            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            mask_path = _find_mask_path(mask_dir, viewpoint.image_name) if mask_dir else None
            if mask_path and os.path.exists(mask_path):
                mask = _load_binary_mask(mask_path, image.shape[1], image.shape[2],
                                         binary_threshold=mask_binary_threshold,
                                         invert=mask_invert).unsqueeze(0)
                mask = mask.expand_as(gt_image)
            else:
                mask = torch.ones_like(gt_image)

            abs_diff = torch.abs(image - gt_image) * mask
            masked_l1 = abs_diff.sum() / (mask.sum() + 1e-8)
            mse = ((image - gt_image) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            masked_psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

            l1_test += masked_l1.item()
            psnr_test += masked_psnr.item()

            if tb_writer and idx < 3:
                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/gt", gt_image[None], iteration)

        l1_test /= len(config['cameras'])
        psnr_test /= len(config['cameras'])
        print(f"[ITER {iteration}] {config['name']}: Masked L1={l1_test:.5f}, PSNR={psnr_test:.3f}")

        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/masked_l1", l1_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/masked_psnr", psnr_test, iteration)


# =========================
# Entry
# =========================
if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS Masked Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--mask_binary_threshold", type=int, default=128)
    parser.add_argument("--mask_invert", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Extract parameters
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    args.save_iterations.append(opt.iterations)
    print(f"Optimizing {dataset.model_path}")

    training(dataset, opt, pipe,
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint,
             args.debug_from,
             mask_dir=args.mask_dir if args.mask_dir else None,
             mask_binary_threshold=args.mask_binary_threshold,
             mask_invert=args.mask_invert)

    print("\nTraining complete.")
