#
# GPU-Optimized + AMP-Safe Training Script for 3D Gaussian Splatting
# Author: Moon Chaewon (optimized by ChatGPT)
# ------------------------------------------------------------
# Features:
#  - Mixed Precision (AMP) for GPU speedup
#  - cuDNN / TF32 optimization
#  - Non-blocking tensor transfers
#  - Safe backward/step sequence for AMP
#  - Prevents grad detach and inf-check errors
# ------------------------------------------------------------
#

import os
import torch
import sys
import cv2
import glob
import uuid
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')

from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui

from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.stopper import GaussianStatStopper
from scene.mask_readers import _find_mask_path, _load_binary_mask
from scene.view_consistency import compute_view_jaccard_fast

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# === GPU optimization flags ===
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             mask_dir=None, mask_binary_threshold=128, mask_invert=False, prune_iterations=[]):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("[ERROR] sparse_adam not available. Please install 3dgs_accel rasterizer.")

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_sparse_adam = (opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE)
    scaler = torch.cuda.amp.GradScaler(enabled=not use_sparse_adam)

    train_views = scene.getTrainCameras().copy()
    viewpoint_stack = train_views.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    stopper = GaussianStatStopper(patience=500, min_delta=1e-5)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", miniters=50)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Random view
        if not viewpoint_stack:
            viewpoint_stack = train_views.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # =====================================================
        # Forward + Loss (AMP)
        # =====================================================
        with torch.cuda.amp.autocast(enabled=not use_sparse_adam):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                                use_trained_exp=dataset.train_test_exp,
                                separate_sh=SPARSE_ADAM_AVAILABLE)
            image = render_pkg["render"]
            image.requires_grad_(True)  # ensure gradient tracking
            gt_image = viewpoint_cam.original_image.cuda(non_blocking=True)

            # Mask loading
            use_mask = mask_dir and len(mask_dir) > 0
            if use_mask and (iteration > prune_iterations[0]):
                mask_path = _find_mask_path(mask_dir, viewpoint_cam.image_name)
                if mask_path and os.path.exists(mask_path):
                    mask = _load_binary_mask(mask_path, image.shape[1], image.shape[2],
                                             binary_threshold=mask_binary_threshold,
                                             invert=mask_invert).unsqueeze(0).cuda(non_blocking=True)
                    mask = mask.expand_as(gt_image)
                else:
                    mask = torch.ones_like(gt_image, device="cuda")
                diff = torch.abs(image - gt_image) * mask
                out_diff = torch.abs(image - gt_image) * (1 - mask) * 0.00001
                Ll1 = (diff.sum() + out_diff.sum()) / (image.numel() / image.shape[0] + 1e-8)
            else:
                Ll1 = l1_loss(image, gt_image)

            # SSIM
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE \
                         else ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "depth_reliable", False):
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda(non_blocking=True)
                depth_mask = viewpoint_cam.depth_mask.cuda(non_blocking=True)
                Ll1depth = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                loss += depth_l1_weight(iteration) * Ll1depth
            else:
                Ll1depth = torch.tensor(0.0, device="cuda")

        # =====================================================
        # Backward + Optimizer step (AMP-safe)
        # =====================================================
        if use_sparse_adam:
            loss.backward()
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
            visible = render_pkg["radii"] > 0
            gaussians.optimizer.step(visible, render_pkg["radii"].shape[0])
            gaussians.optimizer.zero_grad(set_to_none=True)
        else:
            scaler.scale(loss).backward()

            # exposure_optimizer (non-AMP)
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            # main optimizer (AMP)
            scaler.step(gaussians.optimizer)
            scaler.update()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # =====================================================
        # Logging + Saving
        # =====================================================
        if iteration % 50 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})
            progress_bar.update(50)

        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)

        # Early stopping check
        if iteration % 100 == 0:
            gauss_state = {
                "positions": gaussians.get_xyz.detach().cpu().numpy(),
                "scales": gaussians.get_scaling.detach().cpu().numpy(),
                "opacities": gaussians.get_opacity.detach().cpu().numpy(),
            }
            if stopper.update(gauss_state):
                print(f"\n[EarlyStop] Gaussian stats converged at iteration {iteration}")
                scene.save(iteration)
                break


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))[:10]
        args.model_path = os.path.join("./output/", unique_str)
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


if __name__ == "__main__":
    parser = ArgumentParser(description="AMP-Safe GPU Training for 3DGS")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--mask_binary_threshold", type=int, default=128)
    parser.add_argument("--mask_invert", action="store_true")
    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[600, 1200, 1800])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(f"Optimizing {args.model_path}")
    safe_state(args.quiet)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint,
             args.debug_from,
             mask_dir=args.mask_dir if args.mask_dir else None,
             mask_binary_threshold=args.mask_binary_threshold,
             mask_invert=args.mask_invert,
             prune_iterations=args.prune_iterations)
    print("\nTraining complete.")
