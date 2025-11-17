import os
import torch
import clip
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16
from torch.nn.functional import normalize


# ==========================
# 모델 로드
# ==========================
def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def load_dino_model(device):
    model = vit_b_16(weights="IMAGENET1K_V1").to(device)
    model.eval()
    return model


# ==========================
# 이미지 로드 (mask 적용 옵션)
# ==========================
def load_masked_image(img_path, mask_path=None, use_mask=False):
    img = Image.open(img_path).convert("RGB")

    if use_mask and mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L").resize(img.size)
        mask_np = np.array(mask) / 255.0
        img_np = np.array(img).astype(np.float32)
        img_np *= mask_np[..., None]
        img = Image.fromarray(img_np.astype(np.uint8))

    return img


# ==========================
# Feature Extraction
# ==========================
@torch.no_grad()
def extract_clip_feature(model, preprocess, img, device):
    img_t = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(img_t)
    return normalize(feat, dim=-1).cpu().numpy()[0]


@torch.no_grad()
def extract_dino_feature(model, img, device):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(img).unsqueeze(0).to(device)
    feat = model(img_t)
    return normalize(feat, dim=-1).cpu().numpy()[0]


# ==========================
# CLIP + DINO Score 평가
# ==========================
def evaluate_clip_dino(folders, prompt, mask_folder=None, use_mask=False, output_csv="clip_dino_scores.csv", device="cuda"):
    clip_model, clip_preprocess = load_clip_model(device)
    dino_model = load_dino_model(device)

    # Prompt Embedding
    text_feat = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_emb = normalize(clip_model.encode_text(text_feat), dim=-1).cpu().numpy()[0]

    results = []

    for folder in folders:
        folder_name = os.path.basename(folder.rstrip("/"))
        print(f"\n🧩 Evaluating folder: {folder_name}")
        if use_mask:
            print(f"   Applying mask from: {mask_folder}")

        clip_sims, dino_feats = [], []
        img_list = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for img_name in tqdm(img_list, desc=f"{folder_name}"):
            img_path = os.path.join(folder, img_name)
            mask_path = os.path.join(mask_folder, f"frame_{img_name}") if (mask_folder and use_mask) else None

            img = load_masked_image(img_path, mask_path, use_mask)
            
            import matplotlib.pyplot as plt

            # --- for debugging: visualize mask application ---
            # 디버깅용 저장 (최대 5장)
            if use_mask and np.random.rand() < 0.02:
                debug_dir = "../../debug_mask_check"
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"{folder_name}_{img_name}")
                img.save(debug_path)


            # CLIP
            clip_feat = extract_clip_feature(clip_model, clip_preprocess, img, device)
            clip_sims.append(np.dot(clip_feat, text_emb))

            # DINO
            dino_feat = extract_dino_feature(dino_model, img, device)
            dino_feats.append(dino_feat)

        dino_feats = np.stack(dino_feats)
        dino_selfsim = np.mean(np.dot(dino_feats, dino_feats.T))

        results.append({
            "folder": folder_name,
            "clip_mean": np.mean(clip_sims),
            "clip_std": np.std(clip_sims),
            "dino_selfsim": dino_selfsim,
            "mask_applied": use_mask
        })

        print(f"✅ {folder_name}: CLIP={np.mean(clip_sims):.4f}, DINO={dino_selfsim:.4f}, Mask={use_mask}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n📁 Results saved to {output_csv}")
    return df


# ==========================
# Example Run
# ==========================
if __name__ == "__main__":
    # folders_with_mask = [
    #     "../../output_original/lerf_mask/figurines/test/figurines_36/ours_30000/hemisphere_render",
    #     "../../output_mipsplatting_ori/figurines/test_gt/figurines_36/ours_30000/hemisphere_render"
    # ]
    
    # folders_without_mask = [
    #     "../../output_all/lerf_mask/figurines_36/1103_1453/test/ours_30000/hemisphere_render",
    # ]

    # mask_folder = "../../output_mipsplatting_ori/figurines/test_gt/figurines_36/ours_30000/masks"
    # prompt = "a green apple with green leaves"
    # prompt = "a blue elephant figurine"
    # prompt = "a small yellow plastic dog figure with four long legs standing"
    # prompt = "a pink candy package with a yellow cartoon character on it"
    
    prompt = "a transparent box with red "
    
    folders_with_mask = [
        "../../output_original/lerf_mask/figurines/test/figurines_65/ours_30000/hemisphere_render",
        "../../output_mipsplatting_ori/figurines/test_gt/figurines_65/ours_30000/hemisphere_render"
    ]
    
    folders_without_mask = [
        "../../output_all/lerf_mask/figurines_65/1103_1537/test/ours_30000/hemisphere_render",
    ]

    mask_folder = "../../output_mipsplatting_ori/figurines/test_gt/figurines_65/ours_30000/masks"

    

    evaluate_clip_dino(folders_with_mask, prompt, mask_folder, use_mask=True, output_csv="with_mask.csv")

    evaluate_clip_dino(folders_without_mask, prompt, mask_folder, use_mask=False, output_csv="no_mask.csv")
