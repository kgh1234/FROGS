import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import random
import numpy as np
matplotlib.use('Agg')  # for headless environment


from gaussian_renderer import render, network_gui
from scene.mask_readers import _find_mask_path, _load_binary_mask  


# =========================
# Gaussian Overlap Calulation
# =========================

@torch.no_grad()
def gaussian_mask_overlap(xyz, scene, mask_dir, mask_invert=False, iter=0):
    """
    Compute per-Gaussian overlap ratio with 2D binary masks across all training views.

    Args:
        xyz (torch.Tensor): (N, 3) Gaussian centers in world coordinates
        scene (Scene): 3DGS Scene object with camera intrinsics/extrinsics
        mask_dir (str): directory containing GT or binary masks (same name as images)
        mask_invert (bool): if True, invert mask colors (object ↔ background)
        iter (int): current training iteration (for logging/saving)
    Returns:
        overlap_ratio (torch.Tensor): (N,) average overlap ratio across visible views
        avg_mask_ratio (float): average object coverage ratio per view
        overlap_sum (torch.Tensor): (N,) accumulated overlaps
        view_count (torch.Tensor): (N,) number of views that saw each Gaussian
        view_ratios (list[float]): list of mean overlap per view
    """
    views = scene.getTrainCameras()
    n_views = len(views)

    overlap_sum = torch.zeros(xyz.shape[0], device=xyz.device)
    view_count = torch.zeros_like(overlap_sum)

    mask_coverage_all = []
    view_ratios = []

    for v_idx, v in enumerate(views):
        H, W = v.image_height, v.image_width

        mask_path = _find_mask_path(mask_dir, v.image_name)
        if not mask_path:
            continue

        mask = _load_binary_mask(mask_path, H, W, invert=mask_invert).cpu().numpy()
        mask_coverage_all.append(mask.mean())

        uv = v.project_to_screen(xyz)
        u = uv[:, 0].long()
        v_ = uv[:, 1].long()
        valid = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)

        if valid.sum() == 0:
            view_ratios.append(0.0)
            continue

        u_idx_img = u[valid].cpu().numpy()
        v_idx_img = v_[valid].cpu().numpy()
        mask_vals = mask[v_idx_img, u_idx_img]

        overlap_sum[valid] += torch.tensor(mask_vals, device=xyz.device, dtype=torch.float32)
        view_count[valid] += 1.0

        mean_overlap_view = float(np.mean(mask_vals))
        mask_coverage_val = float(mask.mean())
        #print(f"[Overlap@{iter}] View {v_idx:03d}: {mean_overlap_view:.4f} mean overlap, ")

        view_ratios.append(mean_overlap_view)

        # print(f"[Overlap@{iter}] View {v_idx:03d}: "
        #     f"mean_overlap={mean_overlap_view:.4f}, "
        #     f"mask_coverage={mask_coverage_val:.4f}, "
        #     f"valid_gaussians={valid.sum().item()}")

    # ===== Compute final average per-Gaussian overlap =====
    overlap_ratio = overlap_sum / (view_count + 1e-6)
    overlap_ratio[torch.isnan(overlap_ratio)] = 0.0

    avg_mask_ratio = float(np.mean(mask_coverage_all)) if len(mask_coverage_all) > 0 else 0.5

    print(f"[MaskOverlap@{iter}] mean={overlap_ratio.mean():.4f}, "
        f"std={overlap_ratio.std():.4f}, avg_mask_ratio={avg_mask_ratio:.4f}")

    return overlap_ratio, avg_mask_ratio, overlap_sum, view_count, view_ratios




# ==================================================
# View Consistency Filtering (Gaussian Mask Overlap)
# ==================================================
@torch.no_grad()
def gaussian_view_consistency(scene, gaussians, mask_dir, mask_invert=False, threshold=None, save_dir=None, debug_views=None):
    """
    Identify low-outlier (inconsistent) views based on Gaussian–mask overlap and hit ratio.
    Automatically filters low-hit views, saves only those visualizations, and prints lowest 10 hit ratios.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scene.view_consistency import gaussian_mask_overlap, _find_mask_path, _load_binary_mask

    LOW_HIT_THRESHOLD = threshold
    print(f"[GaussianViewConsistency] Checking {len(scene.getTrainCameras())} training views...")

    # === Step 1. Compute global overlap stats ===
    overlap_ratio, avg_mask_ratio, overlap_sum, view_count, view_ratios = gaussian_mask_overlap(
        xyz=gaussians.get_xyz,
        scene=scene,
        mask_dir=mask_dir,
        mask_invert=mask_invert,
        iter=0
    )
    mean_overlaps = np.array(view_ratios, dtype=np.float32)

    if save_dir is None:
        save_dir = os.path.join(scene.model_path, "debug")
    os.makedirs(save_dir, exist_ok=True)

    # === Step 2. Select views to visualize ===
    views = scene.getTrainCameras()
    if debug_views is None:
        debug_views = range(len(views))  # 전체 view 검사
    xyz = gaussians.get_xyz.detach().to(views[0].world_view_transform.device)

    bad_indices, hit_ratios = [], []

    print("[Debug] Visualizing projection alignment (only low-hit views will be saved)...")
    for idx in debug_views:
        if idx >= len(views):
            continue

        cam = views[idx]
        mask_path = _find_mask_path(mask_dir, cam.image_name)
        if not mask_path or not os.path.exists(mask_path):
            print(f"[WARN] View {idx:03d}: mask not found → {cam.image_name}")
            continue

        H, W = cam.image_height, cam.image_width
        mask = _load_binary_mask(mask_path, H, W, invert=mask_invert)
        if mask is None:
            print(f"[WARN] View {idx:03d}: failed to load mask → {cam.image_name}")
            continue
        mask = mask.cpu().numpy()
        h_mask, w_mask = mask.shape[:2]

        # === Project 3D Gaussians to 2D ===
        uv = cam.project_to_screen(xyz)
        u = uv[:, 0].detach().cpu().numpy()
        v = uv[:, 1].detach().cpu().numpy()

        scale_x, scale_y = w_mask / float(W), h_mask / float(H)
        u = np.round(u * scale_x).astype(np.int32)
        v = np.round(v * scale_y).astype(np.int32)
        v = h_mask - v  # flip y-axis (image coordinates)

        valid = (u >= 0) & (u < w_mask) & (v >= 0) & (v < h_mask)
        if valid.sum() == 0:
            continue

        u_valid, v_valid = u[valid], v[valid]
        mask_vals = mask[v_valid, u_valid].astype(np.float32)
        hit_ratio = float(np.mean(mask_vals > 0.5))
        hit_ratios.append((idx, cam.image_name, hit_ratio))

        # === Only visualize low-hit views ===
        # if hit_ratio < LOW_HIT_THRESHOLD:
        #     bad_indices.append(idx)
        #     print(f"[LowHit] View {idx:03d} ({cam.image_name}) → hit_ratio={hit_ratio:.3f} < {LOW_HIT_THRESHOLD}")
        #     fig, ax = plt.subplots(figsize=(6, 5))
        #     ax.imshow(mask, cmap='gray')
        #     ax.scatter(u_valid, v_valid, s=0.5, c='r', alpha=0.3)
        #     ax.set_xlim([0, w_mask])
        #     ax.set_ylim([h_mask, 0])
        #     title = f"{cam.image_name} | hit_ratio={hit_ratio:.3f}"
        #     ax.set_title(title, fontsize=9)
        #     plt.tight_layout()
        #     lowhit_path = os.path.join(save_dir, f"proj_debug_{idx:03d}_LOWHIT.png")
        #     plt.savefig(lowhit_path, dpi=150)
        #     plt.close(fig)
        #     print(f"  [Saved] {lowhit_path}")

    # === Step 3. Summary: top-10 lowest hit ratios ===
    hit_ratios_sorted = sorted(hit_ratios, key=lambda x: x[2])  # sort by hit_ratio
    print("\n[Summary] 🔻 10 lowest hit_ratio views:")
    for rank, (idx, name, hr) in enumerate(hit_ratios_sorted[:10]):
        mark = "⚠️" if hr < LOW_HIT_THRESHOLD else ""
        print(f"  {rank+1:02d}. View {idx:03d} | {name:<25} | hit_ratio={hr:.3f} {mark}")

    # === Step 4. Histogram ===
    # low_hit_vals = [hr for (_, _, hr) in hit_ratios if hr < LOW_HIT_THRESHOLD]
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.hist([hr for (_, _, hr) in hit_ratios], bins=30, color='lightgray', edgecolor='k', alpha=0.6, label="All views")
    # if low_hit_vals:
    #     ax.hist(low_hit_vals, bins=30, color='red', alpha=0.6, label=f"Low hit_ratio (<{LOW_HIT_THRESHOLD})")
    # ax.set_xlabel("Hit Ratio per View")
    # ax.set_ylabel("View Count")
    # ax.set_title("Hit Ratio Distribution (low-hit views in red)")
    # ax.legend()
    # plt.tight_layout()
    # hist_path = os.path.join(save_dir, "view_lowhit_distribution.png")
    # plt.savefig(hist_path, dpi=150)
    # plt.close(fig)
    # print(f"\n[Saved] Low-hit histogram → {hist_path}")
    # print(f"[Done] {len(bad_indices)} low-hit views removed.\n")

    return sorted(set(bad_indices))






#=================================
# Consistency
# =================================

def compute_view_jaccard(scene, gaussians, pipeline, background, threshold=0.2):
    views = scene.getTrainCameras()
    n = len(views)
    visible_sets = []

    for v in views:
        out = render(v, gaussians, pipeline, background)
        vis_mask = out["visibility_filter"] > 0
        visible_ids = torch.nonzero(vis_mask, as_tuple=False).squeeze(-1).cpu().numpy().ravel().tolist()
        visible_sets.append(set(visible_ids))

    jaccard_means = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            inter = len(visible_sets[i] & visible_sets[j])
            union = len(visible_sets[i] | visible_sets[j]) + 1e-6
            sims.append(inter / union)
        mean_sim = sum(sims) / len(sims)
        jaccard_means.append(mean_sim)

    bad_indices = [i for i, score in enumerate(jaccard_means) if score < threshold]
    print(f"[JaccardFilter] {len(bad_indices)}/{n} views flagged (avg sim < {threshold})")

    for i, score in enumerate(jaccard_means):
        if i in bad_indices:
            print(f"* View {i:03d}: {score:.3f} (removed)")
    return bad_indices


def compute_view_jaccard_fast(scene, gaussians, pipeline, background, threshold, sample_k=20):
    views = scene.getTrainCameras()
    n = len(views)
    visible_sets = []

    for v in views:
        out = render(v, gaussians, pipeline, background)
        vis_mask = out["visibility_filter"] > 0
        visible_ids = torch.nonzero(vis_mask, as_tuple=False).squeeze(-1).cpu().numpy().ravel().tolist()
        visible_sets.append(set(visible_ids))

    jaccard_means = []
    for i in range(n):
        sims = []
        sample_idx = random.sample([j for j in range(n) if j != i], min(sample_k, n - 1))
        for j in sample_idx:
            inter = len(visible_sets[i] & visible_sets[j])
            union = len(visible_sets[i] | visible_sets[j]) + 1e-6
            sims.append(inter / union)
        mean_sim = sum(sims) / len(sims)
        jaccard_means.append(mean_sim)

    bad_indices = [i for i, score in enumerate(jaccard_means) if score < threshold]
    print(f"[FastJaccard] {len(bad_indices)}/{n} views flagged (avg sim < {threshold}) [sample_k={sample_k}]")

    return bad_indices