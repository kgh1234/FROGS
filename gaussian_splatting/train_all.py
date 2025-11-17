#
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
import torch
import sys
import cv2
import glob
import uuid
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # for headless environment

from mpl_toolkits.mplot3d import Axes3D
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui


from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.stopper import GaussianStatStopper

from utils.mask_projection_visualization import visualize_mask_projection_with_centers
from scene.view_consistency import compute_view_jaccard, compute_view_jaccard_fast
from scene.view_consistency import gaussian_mask_overlap
from scene.mask_readers import _find_mask_path, _load_binary_mask


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


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             mask_dir=None, mask_binary_threshold=128, mask_invert=False, prune_iterations=[], prune_ratio=1.0):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_views = scene.getTrainCameras().copy()
    viewpoint_stack = train_views.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    opacity_log = []
    shdc_log = []
    iter_log = []


    stopper = GaussianStatStopper(patience=500, min_delta=1e-5)
    print(f"Prune ratio : {prune_ratio}")
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
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)


        with torch.no_grad():

            # --- Opacity 평균 기록 ---
            mean_opacity = gaussians.get_opacity.mean().item()
            opacity_log.append(mean_opacity)

            # --- SH DC(0번째 SH coefficient) 기록 ---
            # gaussians.get_features_dc(): [N,3] 형태일 것
            sh_dc_value = gaussians.get_features_dc.mean().item()
            shdc_log.append(sh_dc_value)

            # --- iteration ---
            iter_log.append(iteration)


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # # filtering on/off
        if iteration == 1000 and iteration < opt.densify_until_iter:
            bad_idx = compute_view_jaccard_fast(scene, gaussians, pipe, background, threshold=0.2)
            if len(bad_idx) > 0:
                train_views = [v for i, v in enumerate(train_views) if i not in bad_idx]
                print(f"[Iter {iteration}] Removed {len(bad_idx)} low-consistency views from training.")
                print(f"[INFO] Remaining training views: {len(train_views)}")

        if iteration == 1801 and iteration < opt.densify_until_iter:
            from scene.view_consistency import gaussian_view_consistency
            bad_idx = gaussian_view_consistency(
                scene=scene,
                gaussians=gaussians,
                mask_dir=mask_dir,
                mask_invert=mask_invert,
                threshold=0.05,       # or fixed like 0.05
            )
            if bad_idx is not None and len(bad_idx) > 0:
                train_views = [v for i, v in enumerate(train_views) if i not in bad_idx]
                removed_views = [v for i, v in enumerate(train_views) if i in bad_idx]
                removed_names = [v.image_name for v in removed_views]
                for i, name in zip(bad_idx, removed_names):
                    print(f"  - View {i:03d}: {name}")
                print(f"[INFO] Remaining training views: {len(train_views)}")
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_views.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        
        use_mask = mask_dir is not None and len(mask_dir) > 0
        if use_mask and (iteration > prune_iterations[0]):
            
            mask_path = _find_mask_path(mask_dir, viewpoint_cam.image_name)
            
            if mask_path and os.path.exists(mask_path):
                mask = _load_binary_mask(mask_path, image.shape[1], image.shape[2],
                                        binary_threshold=mask_binary_threshold,
                                        invert=mask_invert).unsqueeze(0)
                mask = mask.expand_as(gt_image)
            else:
                mask = torch.ones_like(gt_image)
                
            diff = torch.abs(image - gt_image) * mask
            out_diff = torch.abs(image - gt_image) * (1 - mask) * 0.00001
            Ll1 = (diff.sum() + out_diff.sum()) / (image.numel() / image.shape[0] + 1e-8)

            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE \
                         else ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE \
                         else ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)


        # Depth regularization
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

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            from scene.pruning_color import ema_brightness_compensation_from_image

            if not hasattr(gaussians, "_ema_bright_state"):
                gaussians._ema_bright_state = None

            # pruning 시점 또는 주기적 호출 (예: 50 iter마다)
            # if (iteration in prune_iterations):
            #     gaussians._ema_bright_state = ema_brightness_compensation_from_image(
            #         gaussians=gaussians,
            #         render_img=image,      # 현재 view에서 렌더한 결과 그대로 사용
            #          visibility_filter= visibility_filter,
            #         state=gaussians._ema_bright_state,
            #         iteration=iteration,
            #         warmup_iters=500,
            #         luma_momentum=0.95,
            #         tolerance=0.98,
            #         max_global_gain=1.2,
            #         max_step_gain=1.01,
            #         step_alpha=0.2,
            #     )


            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

        

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    if 'prev_brightness' not in locals():
                        prev_brightness = None
                    #prune_iter =[0]
                    prune_iter=[600, 900, 1200]
                    # prune 직전 brightness 저장
                    # if not hasattr(gaussians, "prev_brightness") or gaussians.prev_brightness is None or iteration in prune_iter:
                    #     out = render(viewpoint_cam, gaussians, pipe, background)
                    #     img = out["render"].clamp(0, 1)
                    #     gaussians.prev_brightness = img.mean().item()
                    out = render(viewpoint_cam, gaussians, pipe, background)
                    gaussians.prev_brightness = out["render"].clamp(0,1).mean().item()


                   
                    bad_idx = gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii,
                        mask_dir=mask_dir if use_mask else None,
                        scene=scene,
                        viewpoint_camera=viewpoint_cam,
                        iter=iteration,
                        mask_prune_iter=prune_iter, # pruning on/off
                        prune_ratio=prune_ratio,
                        pipeline=pipe,            
                        background=background,     
                        prev_brightness=prev_brightness 
                    )
                    

                                            
                    
                # from utils.mask_projection_visualization import visualize_mask_pruning_result
                # if use_mask and iteration in prune_iterations:
                #     mask_path = _find_mask_path(mask_dir, viewpoint_cam.image_name)
                #     if mask_path:
                #         visualize_mask_pruning_result(
                #             xyz=gaussians.get_xyz,
                #             viewpoint_cam=viewpoint_cam,
                #             mask_path=mask_path,
                #             prune_mask=getattr(gaussians, "last_prune_mask", None)
                #             if hasattr(gaussians, "last_prune_mask") else None,
                #             invert=False,
                #             save_path=f"{scene.model_path}/debug/mask_prune_vis_iter{iteration}.png"
                # )



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)


            # if iteration % 100 == 0:  
            #     #gaussian_overlap(scene, gaussians, mask_dir, iteration)
            #     gauss_state = {
            #         "positions": gaussians.get_xyz.detach().cpu().numpy(),
            #         "scales": gaussians.get_scaling.detach().cpu().numpy(),
            #         "opacities": gaussians.get_opacity.detach().cpu().numpy(),
            #     }
            #     if stopper.update(gauss_state):
            #         print(f"\n[EarlyStop] Gaussian stats converged at iteration {iteration}")
            #         print(f"[ITER {iteration}] Saving and exiting...")
            #         scene.save(iteration)
            #         return

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(iter_log, opacity_log, label="Opacity Mean")
    plt.plot(iter_log, shdc_log, label="SH-DC Mean")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Opacity & SH-DC Changes Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("brightness_opacity_shdc_curve.png", dpi=200)
    plt.close()


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument('--prune_ratio', type=float, default=1.0)
    

    parser.add_argument('--prune_iterations', nargs="+", type=int, default=[600, 1200, 1800])

    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--mask_binary_threshold", type=int, default=128)
    parser.add_argument("--mask_invert", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
            args.test_iterations, args.save_iterations, 
            args.checkpoint_iterations, args.start_checkpoint, 
            args.debug_from,
            mask_dir=args.mask_dir if args.mask_dir else None,
            mask_binary_threshold=args.mask_binary_threshold,
            mask_invert=args.mask_invert,
            prune_iterations=args.prune_iterations,
            prune_ratio=args.prune_ratio,
            )

    # All done
    print("\nTraining complete.")
