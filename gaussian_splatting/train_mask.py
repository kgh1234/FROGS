# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import re
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

# ===== Optional writers / accel backends =====
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


# =========================
# Mask utilities
# =========================
def _stem(path_or_name: str) -> str:
    """Return filename stem without extension."""
    return os.path.splitext(os.path.basename(path_or_name))[0]

def _find_mask_path(mask_dir: str, image_name_or_path: str):
    """Find a mask file in mask_dir that matches the image stem and has 'mask' optional in name."""
    stem = _stem(image_name_or_path)
    # Try strict: same stem.* in mask_dir
    strict = []
    for ext in ("png", "PNG", "jpg", "JPG", "jpeg", "JPEG", "webp", "WEBP", "bmp", "BMP"):
        p = os.path.join(mask_dir, f"{stem}.{ext}")
        if os.path.isfile(p):
            strict.append(p)
    if strict:
        # Prefer png if multiple
        strict.sort(key=lambda p: (os.path.splitext(p)[1].lower() != ".png", len(p)))
        return strict[0]

    # Fallback: files that start with stem and contain "mask" token
    cands = []
    for ext in ("png", "PNG", "jpg", "JPG", "jpeg", "JPEG", "webp", "WEBP", "bmp", "BMP"):
        cands += glob.glob(os.path.join(mask_dir, f"{stem}*.{ext}"))
    cands = [p for p in cands if "mask" in os.path.basename(p).lower()]
    if not cands:
        return None
    cands.sort(key=lambda p: len(os.path.basename(p)))
    return cands[0]

def _load_binary_mask(mask_path: str, H: int, W: int,
                      binary_threshold: int = 128,
                      invert: bool = False,
                      device: str = "cuda") -> torch.Tensor:
    """
    Load a grayscale mask and convert to binary {0,1} float tensor of shape (H, W).
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if invert:
        m = 255 - m
    m = (m.astype(np.float32) >= float(binary_threshold)).astype(np.float32)
    t = torch.from_numpy(m).to(device)
    return t  # (H, W)


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             mask_dir: str = None, mask_binary_threshold: int = 128, mask_invert: bool = False):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

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

    # ---- Mask cache: image_name -> (H,W) binary tensor ----
    mask_cache = {}  # key: viewpoint_cam.image_name (stem-based) or full name

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Increase SH degree every 1000 iters up to max
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Debug switchover
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Background
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]                        # (3, H, W)
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Optional alpha mask from camera (kept as original project behavior)
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()  # (1 or 3, H, W) or (H,W)
            # normalize to (H,W) then expand to (1,H,W)
            if alpha_mask.ndim == 3 and alpha_mask.shape[0] in (1, 3):
                am = alpha_mask[0] if alpha_mask.shape[0] == 3 else alpha_mask[0]
            else:
                am = alpha_mask
            am = am if am.ndim == 2 else am.squeeze()
            image *= am if am.ndim == 2 else alpha_mask

        # ======== Ground-truth image ========
        gt_image = viewpoint_cam.original_image.cuda()     # (3, H, W)

        # ======== Load and apply our binary object mask for loss ========
        # If mask_dir is provided, we compute **masked losses** so background doesn't drive gradients.
        use_mask = mask_dir is not None and len(mask_dir) > 0
        if use_mask:
            H, W = gt_image.shape[-2:]
            # Build key from the camera's image_name if available; fallback to stem of original path
            key_name = getattr(viewpoint_cam, "image_name", None)
            if key_name is None or not isinstance(key_name, str) or len(key_name) == 0:
                # try to get from the img path recorded in camera metadata if exists
                key_name = _stem(getattr(viewpoint_cam, "image_path", "frame"))
            # Resolve (and cache) mask path/tensor
            if key_name not in mask_cache:
                # Find a candidate mask file
                mask_path = _find_mask_path(mask_dir, key_name)
                if mask_path is None:
                    # Also try using the stem of image_name
                    mask_path = _find_mask_path(mask_dir, _stem(key_name))
                if mask_path is None:
                    # As a last resort, try using the camera's image_name with extension if exposed
                    # (No mask -> use all-ones to avoid crashing)
                    mask_cache[key_name] = torch.ones((H, W), device="cuda", dtype=torch.float32)
                else:
                    mask_cache[key_name] = _load_binary_mask(mask_path, H, W,
                                                             binary_threshold=mask_binary_threshold,
                                                             invert=mask_invert,
                                                             device="cuda")
            mask2d = mask_cache[key_name]  # (H, W) float {0,1}
            mask3 = mask2d.unsqueeze(0).clamp(0.0, 1.0)    # (1, H, W)
            denom = (mask2d.sum() * 3.0).clamp_min(1e-6)

            # ----- Masked L1 -----
            Ll1 = torch.abs((image - gt_image) * mask3).sum() / denom

            # ----- Masked SSIM -----
            # We approximate masked SSIM by feeding masked images to the SSIM function.
            # (Exact masked SSIM would do windowed weighting, but this works well in practice.)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim((image * mask3).unsqueeze(0), (gt_image * mask3).unsqueeze(0))
            else:
                ssim_value = ssim(image * mask3, gt_image * mask3)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        else:
            # Original (unmasked) loss path
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # ======== Depth regularization (unchanged) ========
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Backprop
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                tqdm.set_postfix = getattr(progress_bar, "set_postfix", None)
                if tqdm.set_postfix:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification / pruning (unchanged)
            if iteration < opt.densify_until_iter:
                # Track max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.6, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item() if hasattr(Ll1, "item") else float(Ll1), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # ===== New args for masked training =====
    parser.add_argument("--mask_dir", type=str, default="", help="Directory containing binary masks matching image stems.")
    parser.add_argument("--mask_binary_threshold", type=int, default=128, help="Binarization threshold for mask (0-255).")
    parser.add_argument("--mask_invert", action="store_true", help="Invert mask (use if your convention is swapped).")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        mask_dir=args.mask_dir if args.mask_dir else None,
        mask_binary_threshold=int(args.mask_binary_threshold),
        mask_invert=bool(args.mask_invert)
    )

    # All done
    print("\nTraining complete.")
