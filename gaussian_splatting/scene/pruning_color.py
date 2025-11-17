import torch
import numpy as np

@torch.no_grad()
def auto_brightness_compensation(scene, gaussians, pipeline, background, 
                                 prev_brightness=None, compensate_opacity=True,
                                 compensate_color=True):
    """
    Adaptive brightness equalization after pruning or density change.
    - Measures mean rendered brightness across a few views
    - Scales op acity (and optionally SH colors) to maintain global luminance
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
            print
            
            (f"[AutoComp] SH-DC ×{dc_gain:.2f}")

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




@torch.no_grad()
def auto_brightness_saturation_perview(
    scene, gaussians, pipeline, background,
    viewpoint_cam, state=None,
    luma_momentum=0.9, sat_momentum=0.9,
    luma_tolerance=0.98, sat_tolerance=0.98,
    max_gain_luma=1.4, max_gain_sat=1.3,
    color_gain_gamma=0.85
):
    from gaussian_renderer import render

    # === 안전한 state 초기화 ===
    if state is None:
        state = {'ema_luma_dict': {}, 'ema_sat_dict': {}}
    else:
        if 'ema_luma_dict' not in state:
            state['ema_luma_dict'] = {}
        if 'ema_sat_dict' not in state:
            state['ema_sat_dict'] = {}

    # === 현재 뷰 렌더링 ===
    out = render(viewpoint_cam, gaussians, pipeline, background)
    img = out["render"].clamp(0, 1)

    luma = img.mean().item()
    cstd = img.std(dim=0).mean().item()
    sat = cstd / (luma + 1e-6)

    # === view ID 확보 ===
    vid = getattr(viewpoint_cam, "uid", None)
    if vid is None:
        vid = getattr(viewpoint_cam, "image_name", None)
    if vid is None:
        vid = id(viewpoint_cam)

    # === 기존 값 불러오기 ===
    ema_l = state['ema_luma_dict'].get(vid, luma)
    ema_s = state['ema_sat_dict'].get(vid, sat)

    # === EMA 업데이트 ===
    ema_l = luma_momentum * ema_l + (1 - luma_momentum) * luma
    ema_s = sat_momentum * ema_s + (1 - sat_momentum) * sat

    state['ema_luma_dict'][vid] = ema_l
    state['ema_sat_dict'][vid] = ema_s

    # === 보정 ===
    luma_ratio = luma / (ema_l + 1e-8)
    sat_ratio = sat / (ema_s + 1e-8)

    print(f"[PerViewComp] view={vid} luma_ratio={luma_ratio:.3f}, sat_ratio={sat_ratio:.3f}")

    if luma_ratio < luma_tolerance:
        gain_l = min(1.0 / max(luma_ratio, 1e-4), max_gain_luma)
        if hasattr(gaussians, "_opacity"):
            gaussians._opacity.data *= gain_l
            gaussians._opacity.data.clamp_(0.0, 1.0)
            print(f"[PerViewComp] opacity ×{gain_l:.2f}")

    if sat_ratio < sat_tolerance:
        gain_s = min(1.0 / max(sat_ratio, 1e-4), max_gain_sat)
        gain_s = gain_s ** color_gain_gamma
        if hasattr(gaussians, "_features_rest"):
            gaussians._features_rest.data *= gain_s
            print(f"[PerViewComp] SH-AC ×{gain_s:.2f}")

    return state



import torch
import numpy as np
@torch.no_grad()
def ema_brightness_compensation_from_image(
    gaussians,
    render_img,                 # (3,H,W)
    visibility_filter=None,     # ⭐ 추가: 보이는 Gaussians만 보정
    state=None,                 # dict: {'target_luma','ema_luma','global_gain'}
    iteration=None,
    warmup_iters=1000,
    luma_momentum=0.9,
    tolerance=0.98,
    max_global_gain=1.3,
    max_step_gain=1.03,
    step_alpha=0.2,
):
    """
    ✔ Visibility-aware EMA Brightness Compensation
    - training loop의 render_img만 사용 (추가 렌더 없음)
    - warmup 동안 보정 X (값 안정화)
    - EMA-based global target brightness 유지
    - opacity 보정도 '보이는 가우시안만' 적용
    - 안정성 최우선 (overshoot 방지)
    """

    # --- 초기 상태 저장 ---
    if state is None:
        state = {"target_luma": None, "ema_luma": None, "global_gain": 1.0}

    img = render_img.clamp(0, 1)
    curr_luma = img.mean().item()

    # --- 초기 1회 실행 ---
    if state["target_luma"] is None:
        state["target_luma"] = curr_luma
        state["ema_luma"] = curr_luma
        return state

    # --- EMA 업데이트 ---
    prev_ema = state["ema_luma"]
    ema_now = luma_momentum * prev_ema + (1 - luma_momentum) * curr_luma
    state["ema_luma"] = ema_now

    target = state["target_luma"]

    # --- warmup: target만 천천히 유지 ---
    if iteration is not None and iteration < warmup_iters:
        state["target_luma"] = 0.99 * target + 0.01 * ema_now
        return state

    # --- 밝기 비율 계산 ---
    luma_ratio = ema_now / (target + 1e-8)

    # 충분히 밝으면 보정 불필요
    if luma_ratio >= tolerance:
        state["target_luma"] = 0.995 * target + 0.005 * ema_now
        return state

    # --- 이상적 gain 계산 ---
    ideal_gain = 1.0 / max(luma_ratio, 1e-6)

    # 전체 gain을 제한
    ideal_gain = min(ideal_gain, max_global_gain / state["global_gain"])

    # --- 작은 step만 적용 ---
    step_gain = ideal_gain ** step_alpha
    step_gain = min(step_gain, max_step_gain)

    # 효과가 매우 작으면 skip
    if step_gain <= 1.001:
        return state

    # ----------------------------------------------
    # ⭐ 핵심: visibility_filter로 현재 보이는 가우시안만 보정
    # ----------------------------------------------
    if hasattr(gaussians, "_opacity"):
        if visibility_filter is not None:
            gaussians._opacity.data[visibility_filter] *= step_gain
        else:
            # fallback (무조건 전부 보정)
            gaussians._opacity.data *= step_gain

        gaussians._opacity.data.clamp_(0.0, 1.0)
        state["global_gain"] *= step_gain

        print(f"[EMA-Bright] visible-opacity ×{step_gain:.3f}, global_gain={state['global_gain']:.3f}")

    return state

@torch.no_grad()
def prune_shdc_global_brightness_compensation(
    scene, gaussians, pipeline, background,
    target_views=4           # brightness 측정할 뷰 수
):
    """
    Pruning 직후 한 번만 실행하는 안전한 SH-DC brightness compensation.
    - opacity는 건드리지 않음
    - SH DC만 global scale
    - multi-view 안정성 보장
    """

    from gaussian_renderer import render

    # ---- 1. 몇 개 view에서 평균 brightness 계산 ----
    cams = scene.getTrainCameras()[:target_views]
    brightness_vals = []

    for cam in cams:
        out = render(cam, gaussians, pipeline, background)
        img = out["render"].clamp(0, 1)
        brightness_vals.append(img.mean().item())

    curr_brightness = float(np.mean(brightness_vals))

    # ---- 2. Target brightness: prune 이전 평균 밝기 ----
    # prune 전 brightness를 저장해둔 값이 필요함
    # GaussianModel 안에 prev_brightness가 들어있다고 가정
    target_brightness = getattr(gaussians, "prev_brightness", None)

    if target_brightness is None:
        # prune 전 brightness 정보가 없다면 보정 생략
        print("[SHDC-Bright] No previous brightness stored → skip")
        return

    # ---- 3. scaling factor 계산 ----
    gain = target_brightness / (curr_brightness + 1e-8)

    # 너무 큰 gain은 clamp
    #gain = float(np.clip(gain, 0.7, 1.3))

    print(f"[SHDC-Bright] curr={curr_brightness:.4f}, target={target_brightness:.4f}, gain={gain:.3f}")

    # ---- 4. SH-DC만 scale ----
    if hasattr(gaussians, "_features_dc"):
        gaussians._features_dc.data *= gain
        gaussians._features_dc.data.clamp_(-3.0, 3.0)
        print(f"[SHDC-Bright] Applied SH-DC ×{gain:.3f}")
    else:
        print("[SHDC-Bright] Warning: gaussians._features_dc not found")


@torch.no_grad()
def prune_shdc_global_brightness_compensation_ema(
    scene, gaussians, pipeline, background,
    state=None,
    target_views=6,
    ema_momentum=0.95,
    gain_max=1.3,
):
    """
    EMA 기반 pruning 직후 SH-DC 보정 (안전/권장)
    - opacity는 절대 수정하지 않음
    - SH-DC만 EMA 기반으로 완만하게 보정
    - gain < 1(어두워지게 되는 경우)은 skip하여 안전성 확보
    """

    from gaussian_renderer import render

    # ---- state 구조 초기화 ----
    if state is None:
        state = {"ema_brightness": None, "target_brightness": None}

    # ---- 1. prune 이후 현재 brightness 계산 ----
    cams = scene.getTrainCameras()[:target_views]
    brightness_vals = []

    for cam in cams:
        out = render(cam, gaussians, pipeline, background)
        img = out["render"].clamp(0, 1)
        brightness_vals.append(img.mean().item())

    curr_brightness = float(np.mean(brightness_vals))

    # ---- 2. prune 이전 brightness (target) ----
    target_brightness = getattr(gaussians, "prev_brightness", None)
    if target_brightness is None:
        print("[SHDC-EMA] No previous brightness stored → skip")
        return state

    # ---- 3. EMA 업데이트 ----
    if state["ema_brightness"] is None:
        state["ema_brightness"] = curr_brightness
    else:
        state["ema_brightness"] = (
            ema_momentum * state["ema_brightness"]
            + (1 - ema_momentum) * curr_brightness
        )

    ema_curr = state["ema_brightness"]

    # ---- 4. gain 계산 (EMA 기반) ----
    gain = target_brightness / (ema_curr + 1e-8)

    # gain < 1.0 → 밝기 낮추려는 방향이므로 skip (안정성 필수)
    if gain < 1.0:
        print(f"[SHDC-EMA] gain={gain:.3f} < 1.0 → skip darkening")
        return state

    # clamp
    gain = min(gain, gain_max)

    print(
        f"[SHDC-EMA] curr={curr_brightness:.4f}, "
        f"EMA={ema_curr:.4f}, target={target_brightness:.4f}, "
        f"gain={gain:.3f}"
    )

    # ---- 5. SH-DC만 보정 ----
    if hasattr(gaussians, "_features_dc"):
        gaussians._features_dc.data *= gain
        gaussians._features_dc.data.clamp_(-3.0, 3.0)
        print(f"[SHDC-EMA] Applied SH-DC ×{gain:.3f}")
    else:
        print("[SHDC-EMA] Warning: _features_dc not found")

    return state
