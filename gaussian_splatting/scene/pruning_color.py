import torch
import numpy as np

@torch.no_grad()
def auto_brightness_compensation(scene, gaussians, pipeline, background, 
                                 prev_brightness=None, compensate_opacity=True,
                                 compensate_color=True):
    """
    Adaptive brightness equalization after pruning or density change.
    - Measures mean rendered brightness across a few views
    - Scales opacity (and optionally SH colors) to maintain global luminance
    """

    # Lazy import to avoid circular dependency
    from gaussian_renderer import render  

    # === Step 1. 측정 ===
    views = scene.getTrainCameras()[:4]
    brightness_vals = []
    for v in views:
        out = render(v, gaussians, pipeline, background)
        img = out["render"].clamp(0, 1)
        brightness_vals.append(img.mean().item())

    curr_brightness = np.mean(brightness_vals)

    if prev_brightness is None:
        print(f"[AutoBright] init brightness={curr_brightness:.4f}")
        return curr_brightness

    # === Step 2. 비율 계산 ===
    ratio = curr_brightness / (prev_brightness + 1e-8)
    print(f"[AutoBright] brightness ratio={ratio:.3f} (curr={curr_brightness:.4f}, prev={prev_brightness:.4f})")

    # === Step 3. Opacity 보정 ===
    if compensate_opacity and ratio < 0.98:
        scale = 1.0 / ratio
        if hasattr(gaussians, "_opacity"):
            gaussians._opacity.data *= scale
            gaussians._opacity.data.clamp_(0.0, 1.0)
            print(f"[AutoBright] opacity scaled ×{scale:.2f}")
        else:
            print("[AutoBright] Warning: _opacity not found")

    # === Step 4. SH 색상 보정 ===
    if compensate_color and ratio < 0.98:
        if hasattr(gaussians, "_features_dc"):
            gaussians._features_dc.data *= (1.0 / ratio)
            gaussians._features_dc.data.clamp_(-3.0, 3.0)
            print(f"[AutoBright] SH DC features boosted ×{1.0/ratio:.2f}")

    return curr_brightness



@torch.no_grad()
def auto_brightness_saturation_compensation(
    scene, gaussians, pipeline, background,
    state=None,                 # dict: {'ema_luma':..., 'ema_sat':...}
    luma_momentum=0.9,          # EMA for stability
    sat_momentum=0.9,
    luma_tolerance=0.98,        # only compensate if curr/target < 0.98
    sat_tolerance=0.98,
    max_gain_luma=1.4,          # safety limits
    max_gain_sat=1.3,
    color_gain_gamma=0.85,      # soften color gain (power < 1)
    clamp_dc=3.0,               # SH-DC clamp
    clamp_opacity=(0.0, 1.0)    # opacity clamp
):
    """
    Brightness (luma) + Saturation (chroma) compensation with EMA stabilization.
    - Luma 보정: opacity와 SH-DC를 적당히 스케일
    - Saturation 보정: SH-AC(features_rest) 에너지를 스케일
    - state(dict)에 EMA 타깃 저장/업데이트
    """
    # Lazy import (avoid circular)
    from gaussian_renderer import render

    if state is None:
        state = {}

    # === 1) Render 몇 뷰 평균으로 luma / sat 측정 ===
    views = scene.getTrainCameras()[:4]
    lumas, sats = [], []
    for v in views:
        out = render(v, gaussians, pipeline, background)
        img = out["render"].clamp(0, 1)  # [3,H,W]
        # luma (mean brightness)
        luma = img.mean().item()

        # saturation proxy: (per-pixel channel std) / (per-pixel mean+eps)
        # 평균값으로 단순화: 채널표준편차의 평균 / 평균휘도
        cstd = img.std(dim=0).mean().item()         # 평균 채널 std
        mean_luma = img.mean().item() + 1e-6
        sat = (cstd / mean_luma)

        lumas.append(luma)
        sats.append(sat)

    curr_luma = float(np.mean(lumas))
    curr_sat  = float(np.mean(sats))

    # === 2) EMA 타깃 업데이트 ===
    if "ema_luma" not in state:
        state["ema_luma"] = curr_luma
    else:
        state["ema_luma"] = luma_momentum * state["ema_luma"] + (1 - luma_momentum) * curr_luma

    if "ema_sat" not in state:
        state["ema_sat"] = curr_sat
    else:
        state["ema_sat"] = sat_momentum * state["ema_sat"] + (1 - sat_momentum) * curr_sat

    target_luma = state["ema_luma"]
    target_sat  = state["ema_sat"]

    # === 3) Luma 보정 (opacity + SH-DC)
    luma_ratio = curr_luma / (target_luma + 1e-8)
    print(f"[AutoComp] Luma ratio={luma_ratio:.3f} (curr={curr_luma:.4f}, ema={target_luma:.4f})")

    if luma_ratio < luma_tolerance:
        gain_l = min(1.0 / max(luma_ratio, 1e-4), max_gain_luma)

        # opacity 우선 보정
        if hasattr(gaussians, "_opacity"):
            gaussians._opacity.data *= gain_l
            gaussians._opacity.data.clamp_(*clamp_opacity)
            print(f"[AutoComp] opacity ×{gain_l:.2f}")

        # 잔여 오차가 있으면 DC도 살짝 보정
        if hasattr(gaussians, "_features_dc"):
            dc_gain = gain_l ** 0.5  # DC는 절반 강도로
            gaussians._features_dc.data *= dc_gain
            gaussians._features_dc.data.clamp_(-clamp_dc, clamp_dc)
            print(f"[AutoComp] SH-DC ×{dc_gain:.2f}")

    # === 4) Saturation 보정 (SH-AC / features_rest)
    sat_ratio = curr_sat / (target_sat + 1e-8)
    print(f"[AutoComp] Sat ratio={sat_ratio:.3f} (curr={curr_sat:.4f}, ema={target_sat:.4f})")

    if sat_ratio < sat_tolerance:
        gain_s = min(1.0 / max(sat_ratio, 1e-4), max_gain_sat)
        gain_s = gain_s ** color_gain_gamma  # 부드럽게
        if hasattr(gaussians, "_features_rest") and gaussians._features_rest.numel() > 0:
            gaussians._features_rest.data *= gain_s
            # 과도한 컬러 링 방지: 전체 에너지 클램프 (옵션)
            # (원하면 여기에 norm 기반 에너지 제한 추가 가능)
            print(f"[AutoComp] SH-AC ×{gain_s:.2f}")

    return state
