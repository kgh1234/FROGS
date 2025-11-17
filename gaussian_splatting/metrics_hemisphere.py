import os
import cv2
import csv
import torch
import numpy as np
from utils.loss_utils import ssim as torch_ssim
from lpips import LPIPS


# ==========================================
# Metric Functions
# ==========================================
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def calculate_ssim_torch(img1, img2):
    """3DGS-style SSIM using torch tensor (expects B,C,H,W, range 0~1)."""
    img1_t = torch.tensor(img1 / 255.0).permute(2, 0, 1).unsqueeze(0)
    img2_t = torch.tensor(img2 / 255.0).permute(2, 0, 1).unsqueeze(0)
    return torch_ssim(img1_t, img2_t).item()


# ==========================================
# Main Comparison Function
# ==========================================
def compare_metrics_every_3(folderA, folderB, device="cuda"):
    print("=" * 90)
    print(f"[비교 시작] {folderA.split('/')[-1]}  ↔  {folderB.split('/')[-1]}")
    print("=" * 90)

    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgsA = sorted([f for f in os.listdir(folderA) if f.lower().endswith(exts)])
    imgsB = sorted([f for f in os.listdir(folderB) if f.lower().endswith(exts)])

    if len(imgsA) != len(imgsB):
        print(f"[경고] 두 폴더의 이미지 개수가 다릅니다: {len(imgsA)} vs {len(imgsB)}")
        n = min(len(imgsA), len(imgsB))
        imgsA, imgsB = imgsA[:n], imgsB[:n]

    lpips_model = LPIPS(net='vgg').to(device)

    results = []  # CSV 저장용

    psnr_list, ssim_list, lpips_list = [], [], []

    for i in range(0, len(imgsA), 3):
        pathA = os.path.join(folderA, imgsA[i])
        pathB = os.path.join(folderB, imgsB[i])

        imgA = cv2.imread(pathA)
        imgB = cv2.imread(pathB)

        if imgA is None or imgB is None:
            print(f"[스킵] {imgsA[i]} 또는 {imgsB[i]} 읽기 실패")
            continue

        if imgA.shape != imgB.shape:
            imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

        # === PSNR ===
        psnr_val = calculate_psnr(imgA, imgB)

        # === SSIM (torch-based) ===
        ssim_val = calculate_ssim_torch(imgA, imgB)

        # === LPIPS ===
        imgA_rgb = imgA[..., ::-1].copy()
        imgB_rgb = imgB[..., ::-1].copy()

        imgA_t = torch.tensor(imgA_rgb).permute(2, 0, 1).unsqueeze(0).float()
        imgB_t = torch.tensor(imgB_rgb).permute(2, 0, 1).unsqueeze(0).float()
        imgA_t = (imgA_t / 255.0) * 2 - 1
        imgB_t = (imgB_t / 255.0) * 2 - 1
        lpips_val = lpips_model(imgA_t.to(device), imgB_t.to(device)).item()

        # === inf 제외 조건 ===
        if np.isinf(psnr_val):
            print(f"[제외] {imgsA[i]} → PSNR=inf (완전 동일 이미지)")
            continue

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

        results.append({
            "index": i,
            "image": imgsA[i],
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "LPIPS": lpips_val
        })

        print(f"[{i:03d}] {imgsA[i]} → PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

    # ===========================
    # CSV 저장
    # ===========================
    save_name = f"metrics_result_{os.path.basename(folderB)}.csv"
    with open(save_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image", "PSNR", "SSIM", "LPIPS"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📁 결과 CSV 저장 완료 → {save_name}")

    # ===========================
    # 평균 출력 (inf 제외된 값들만)
    # ===========================
    if psnr_list:
        print("\n===== 평균 결과 (3의 배수 인덱스만, inf 제외) =====")
        print(f"📊 PSNR 평균 : {np.mean(psnr_list):.2f} dB")
        print(f"📊 SSIM 평균 : {np.mean(ssim_list):.4f}")
        print(f"📊 LPIPS 평균: {np.mean(lpips_list):.4f}")
    else:
        print("비교 가능한 유효 이미지가 없습니다.")


# ==========================================
# 실행 예시
# ==========================================
GT   = '../../output_mipsplatting_ori/figurines/test/figurines_36/ours_30000/images'
GS   = '../../output_original/lerf_mask/figurines/test/figurines_36/ours_30000/images'
OURS = '../../output_all/lerf_mask/figurines_36/1025_0312/test/ours_20241/images'

#compare_metrics_every_3(GT, GS)
compare_metrics_every_3(GT, OURS)
