import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # for headless environment


from gaussian_renderer import render, network_gui
from FROGS.gaussian_splatting.scene.mask_readers import _find_mask_path, _load_binary_mask  


# =========================
# View Consistency Filtering
# =========================
@torch.no_grad()
def gaussian_overlap(scene, gaussians, mask_dir, iteration, num_views=3, mask_invert=False):
    
    if mask_dir is None or not os.path.exists(mask_dir):
        print(f"[Vis] mask_dir not found → skip visualization")
        return

    xyz = gaussians.get_xyz.detach()  # (N,3)
    xyz_np = xyz.cpu().numpy()

    overlap_sum = torch.zeros(xyz.shape[0], device=xyz.device)
    view_count = torch.zeros_like(overlap_sum)

    views = scene.getTrainCameras()[:]
    save_dir = os.path.join(scene.model_path, f"mask_debug_iter{iteration}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Vis] === Projection-aligned Gaussian–Mask Overlap (iter={iteration}) ===")

    for v in views:
        H, W = v.image_height, v.image_width
        mask_path = _find_mask_path(mask_dir, v.image_name)
        if not mask_path:
            print(f"[Vis] No mask found for {v.image_name}")
            continue

        mask = _load_binary_mask(mask_path, H, W, invert=mask_invert).cpu().numpy()
        print(f"[MaskDebug] {v.image_name} → {os.path.basename(mask_path)} | mean={mask.mean():.3f}")

        uv = v.project_to_screen(xyz)
        u = uv[:, 0].long()
        v_ = uv[:, 1].long()

        valid = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)
        if valid.sum() == 0:
            continue

        u_idx = u[valid].cpu().numpy()
        v_idx = v_[valid].cpu().numpy()
        mask_vals = mask[v_idx, u_idx]

        overlap_sum[valid] += torch.tensor(mask_vals, device=xyz.device)
        view_count[valid] += 1.0

    # === Normalize overlap ===
    overlap_ratio = overlap_sum / (view_count + 1e-6)
    overlap_np = overlap_ratio.cpu().numpy()
    nonzero_mask = overlap_np > 0

    print(f"[Iter {iteration}] Nonzero overlap ratio: {nonzero_mask.sum()}/{len(overlap_np)} = {nonzero_mask.mean():.4f}")
    print(f"[Iter {iteration}] Overlap stats → min={overlap_np.min():.4f}, max={overlap_np.max():.4f}, mean={overlap_np.mean():.4f}")

    # === 3D Heatmap ===
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2],
        c=np.log1p(overlap_np), s=3, cmap='jet'
    )
    plt.colorbar(sc, ax=ax, pad=0.05)
    ax.set_title(f"Gaussian Overlap Heatmap (iter={iteration})")
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, f"overlap_heatmap_{iteration}.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    print(f"[Vis] Saved normalized overlap heatmap → {heatmap_path}")

    # === Per-view summary ===
    print("\n[Vis] === Per-view overlap summary ===")
    for v in views:
        H, W = v.image_height, v.image_width
        mask_path = _find_mask_path(mask_dir, v.image_name)
        if not mask_path:
            continue
        mask = _load_binary_mask(mask_path, H, W, invert=mask_invert).cpu().numpy()
        uv = v.project_to_screen(xyz)
        u = uv[:, 0].long()
        v_ = uv[:, 1].long()
        valid = (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)
        if valid.sum() == 0:
            continue
        u_idx = u[valid].cpu().numpy()
        v_idx = v_[valid].cpu().numpy()
        mask_vals_np = mask[v_idx, u_idx].astype(np.float32)
        overlaps_for_view = overlap_np[valid.cpu().numpy()]
        selected = mask_vals_np > 0.5
        mean_overlap = overlaps_for_view[selected].mean() if np.any(selected) else 0.0
        print(f"{v.image_name:<25s} | mean overlap = {mean_overlap:.4f}")





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

