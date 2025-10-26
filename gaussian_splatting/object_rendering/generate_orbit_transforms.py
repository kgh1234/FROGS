#!/usr/bin/env python3
"""
PLY 중심을 기준으로 360° 오빗 카메라 경로를 생성하고
3DGS/Blender-호환 transforms JSON을 저장한다.

기능:
- forward = -Z (NeRF/3DGS 규약) 정렬 옵션 (--nerf_negz)
- camera_angle_x, camera_angle_y 자동 계산
- frames[*].file_path 연속 파일명 자동 생성
  * 확장자 포함/미포함 저장 (--store_no_ext)
  * 절대(--absolute)/상대(--relative) 경로 저장
- 이미지 존재 여부 검증 (--verify)
- 상하 반전 보정(카메라 롤 180°) (--roll180)

렌더 시 file_path를 확장자 없이 저장했다면:
python render.py ... --images .jpg
"""

import os
import json
import math
import argparse
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


# ---------------- I/O ----------------
def load_ply_points(path: str) -> np.ndarray:
    if o3d is None:
        raise RuntimeError("open3d is required. Install: pip install open3d")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY not found: {path}")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("No points found in the PLY.")
    return pts


# ------------- Center & Radius -------------
def compute_center(points: np.ndarray, method: str = "median") -> np.ndarray:
    if method == "median":
        return np.median(points, axis=0)
    elif method == "mean":
        return np.mean(points, axis=0)
    raise ValueError("method must be 'median' or 'mean'")


def auto_radius(points: np.ndarray, scale: float = 0.6) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = np.linalg.norm(maxs - mins)
    r = (0.5 * diag) * scale
    return float(max(r, 1e-3))


# ---------------- Intrinsics ----------------
def focal_from_fov(fov_deg: float, img_w: int) -> float:
    """Horizontal FOV(deg) -> fx (pixels)"""
    fov_rad = math.radians(fov_deg)
    return (img_w / 2.0) / math.tan(fov_rad / 2.0)


# ---------------- Pose builders ----------------
def look_at_c2w(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    camera-to-world (c2w): columns = [right, up, forward, eye]
    마지막 행은 [0,0,0,1]. 기본적으로 +Z가 center 향함.
    """
    eye = np.asarray(eye, float)
    center = np.asarray(center, float)
    up = np.asarray(up, float)

    forward = center - eye
    n = np.linalg.norm(forward)
    if n < 1e-8:
        raise ValueError("eye == center (no view direction).")
    forward /= n

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        tmp = np.array([0.0, 0.0, 1.0], float)
        right = np.cross(forward, tmp)
        rn = np.linalg.norm(right)
        if rn < 1e-8:
            tmp = np.array([1.0, 0.0, 0.0], float)
            right = np.cross(forward, tmp)
            rn = np.linalg.norm(right)
    right /= rn
    true_up = np.cross(right, forward)

    c2w = np.eye(4, dtype=float)
    c2w[0:3, 0] = right
    c2w[0:3, 1] = true_up
    c2w[0:3, 2] = forward
    c2w[0:3, 3] = eye
    return c2w


def flip_to_nerf_negz(c2w: np.ndarray) -> np.ndarray:
    """
    NeRF/3DGS 규약: 카메라는 -Z 바라봄.
    손잡이성 보존 위해 right(X, col0)와 forward(Z, col2) 동시에 부호 반전.
    """
    M = c2w.copy()
    M[0:3, 0] *= -1.0  # right
    M[0:3, 2] *= -1.0  # forward
    return M


def apply_roll180(c2w: np.ndarray) -> np.ndarray:
    """
    상하 뒤집힘 보정: 시선축(+Z)은 그대로 두고 right(+X), up(+Y)만 부호 반전.
    (카메라 롤 180°)
    """
    M = c2w.copy()
    M[0:3, 0] *= -1.0  # right
    M[0:3, 1] *= -1.0  # up
    return M


def generate_orbit_c2w(center: np.ndarray,
                       radius: float,
                       num_views: int,
                       height: float,
                       up: np.ndarray,
                       nerf_negz: bool = False) -> list:
    poses = []
    for i in range(num_views):
        theta = (2.0 * math.pi) * (i / num_views)
        ex = center[0] + radius * math.cos(theta)
        ey = center[1] + height
        ez = center[2] + radius * math.sin(theta)
        c2w = look_at_c2w(np.array([ex, ey, ez], float), center, up)
        if nerf_negz:
            c2w = flip_to_nerf_negz(c2w)
        poses.append(c2w)
    return poses


# ---------------- Export ----------------
def export_json(out_path: str,
                c2w_list: list,
                img_w: int, img_h: int,
                fx: float, fy: float, cx: float, cy: float,
                file_dir: str, prefix: str, ext: str,
                pad: int, start: int,
                relative: bool, verify: bool,
                store_no_ext: bool = False,
                absolute: bool = False,
                extra: dict = None) -> str:
    """
    absolute=True 가 가장 우선. (absolute -> relative -> raw path)
    store_no_ext=True 이면 file_path를 확장자 없이 저장.
    """
    camera_angle_x = 2.0 * math.atan(img_w / (2.0 * fx))
    camera_angle_y = 2.0 * math.atan(img_h / (2.0 * fy))

    data = {
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(img_w),
        "h": int(img_h),
        "camera_angle_x": float(camera_angle_x),
        "camera_angle_y": float(camera_angle_y),
        "camera_model": "OPENCV",
        "frames": [],
    }
    if extra:
        data.update(extra)

    out_p = Path(out_path)
    base = out_p.parent
    img_dir = Path(file_dir).resolve()
    if not img_dir.exists():
        raise FileNotFoundError(f"images dir not found: {img_dir}")

    for i, c2w in enumerate(c2w_list):
        idx = start + i
        stem = f"{prefix}{idx:0{pad}d}"

        # 검증은 확장자 포함 파일로
        fverify = img_dir / f"{stem}.{ext}"
        if verify and not fverify.exists():
            print(f"⚠️ missing: {fverify}")

        # 저장 경로는 옵션에 따라
        fpath = img_dir / stem if store_no_ext else img_dir / f"{stem}.{ext}"

        if absolute:
            file_path = str(fpath.resolve())
        elif relative:
            file_path = os.path.relpath(fpath, start=base)
        else:
            file_path = str(fpath)

        data["frames"].append({
            "file_path": file_path,
            "transform_matrix": c2w.tolist(),
        })

    os.makedirs(base, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved JSON: {out_p}")
    print(f"📸 File path mode: {'absolute' if absolute else ('relative' if relative else 'raw')}")
    print(f"🔖 file_path contains extension: {'NO (store_no_ext=True)' if store_no_ext else 'YES'}")
    return str(out_p)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # inputs
    ap.add_argument("--ply", required=True, help="point_cloud.ply")
    ap.add_argument("--out", required=True, help="output JSON (e.g., transforms_train.json)")
    # orbit
    ap.add_argument("--num_views", type=int, default=36)
    ap.add_argument("--height", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=None, help="if omitted, auto by bbox diagonal")
    ap.add_argument("--up", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    ap.add_argument("--use_mean", action="store_true", help="use mean center instead of median")
    ap.add_argument("--nerf_negz", action="store_true", help="flip to NeRF/3DGS convention (forward=-Z)")
    ap.add_argument("--roll180", action="store_true", help="상하 뒤집힘 보정(카메라 롤 180°)")
    # intrinsics
    ap.add_argument("--img_w", type=int, default=1920)
    ap.add_argument("--img_h", type=int, default=1080)
    ap.add_argument("--fov_deg", type=float, default=60.0, help="used if fx/fy are not provided")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    # file_path sequence
    ap.add_argument("--images_dir", required=True, help="directory containing images")
    ap.add_argument("--prefix", default="frame_", help="filename prefix (default: frame_)")
    ap.add_argument("--ext", default="png", help="extension WITHOUT dot (e.g., png/jpg)")
    ap.add_argument("--pad", type=int, default=5, help="zero padding (default: 5)")
    ap.add_argument("--start", type=int, default=1, help="start index (default: 1)")
    ap.add_argument("--relative", action="store_true", help="store file_path relative to OUT directory")
    ap.add_argument("--absolute", action="store_true", help="store file_path as absolute path")
    ap.add_argument("--verify", action="store_true", help="warn if files are missing")
    ap.add_argument("--store_no_ext", action="store_true", help="store file_path WITHOUT extension")
    args = ap.parse_args()

    # sanity for path mode
    if args.absolute and args.relative:
        print("⚠️ Both --absolute and --relative set. Using --absolute precedence.")

    # load points & center/radius
    pts = load_ply_points(args.ply)
    center = compute_center(pts, method="mean" if args.use_mean else "median")
    radius = auto_radius(pts, scale=0.6) if args.radius is None else float(args.radius)

    # intrinsics
    if args.fx is not None and args.fy is not None:
        fx, fy = float(args.fx), float(args.fy)
    else:
        fx = focal_from_fov(args.fov_deg, args.img_w)
        fy = fx * (args.img_h / args.img_w)
    cx = float(args.cx) if args.cx is not None else (args.img_w / 2.0)
    cy = float(args.cy) if args.cy is not None else (args.img_h / 2.0)

    # poses
    c2w_list = generate_orbit_c2w(
        center=center,
        radius=radius,
        num_views=args.num_views,
        height=args.height,
        up=np.array(args.up, float),
        nerf_negz=args.nerf_negz
    )

    # 상하 뒤집힘 보정(필요 시 롤 180°)
    if args.roll180:
        c2w_list = [apply_roll180(M) for M in c2w_list]

    # export
    out_path = export_json(
        out_path=args.out,
        c2w_list=c2w_list,
        img_w=args.img_w, img_h=args.img_h,
        fx=fx, fy=fy, cx=cx, cy=cy,
        file_dir=args.images_dir, prefix=args.prefix, ext=args.ext,
        pad=args.pad, start=args.start,
        relative=(False if args.absolute else args.relative),
        verify=args.verify,
        store_no_ext=args.store_no_ext,
        absolute=args.absolute,
        extra={
            "note": "Orbit cameras around PLY center",
            "point_cloud_path": args.ply,
            "orbit_radius": radius,
            "height": args.height,
            "num_views": args.num_views,
            "center_used": "mean" if args.use_mean else "median",
            "center_values": center.tolist(),
        }
    )

    # summary
    print("=== Summary ===")
    print(f"Points           : {pts.shape[0]:,}")
    print(f"Center           : {center.tolist()} ({'mean' if args.use_mean else 'median'})")
    print(f"Orbit radius     : {radius:.4f}")
    print(f"Intrinsics       : fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, w={args.img_w}, h={args.img_h}")
    print(f"camera_angle_x   : {2.0 * math.atan(args.img_w / (2.0 * fx)):.6f} rad")
    print(f"camera_angle_y   : {2.0 * math.atan(args.img_h / (2.0 * fy)):.6f} rad")
    print(f"Saved JSON       : {out_path}")
    print(f"Paths mode       : {'absolute' if args.absolute else ('relative' if args.relative else 'raw')}")
    print(f"Extension saved  : {'NO (store_no_ext=True; use --images at render time)' if args.store_no_ext else 'YES'}")
    print(f"NeRF -Z forward  : {'ON' if args.nerf_negz else 'OFF'}")
    print(f"Roll 180 applied : {'YES' if args.roll180 else 'NO'}")


if __name__ == "__main__":
    main()
