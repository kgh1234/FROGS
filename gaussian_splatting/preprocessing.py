import os, cv2, numpy as np
from pathlib import Path

ROOT = '../../masked_datasets/mipnerf'

scenelist = os.listdir(ROOT)

for scene in scenelist:
    scene_dir = Path(os.path.join(ROOT, scene))
    scene_dir = Path('../../masked_datasets/mipnerf/garden')

    if not scene_dir.is_dir():
        print("not a directory:", scene)
        continue
    

    ori_dir = scene_dir / "images_ori"
    mask_dir = scene_dir / "mask"
    out_dir = scene_dir / "images"
    
    if ori_dir.exists():
        print("already exists:", scene)
        continue
    os.rename(scene_dir / "images", scene_dir / "images_ori")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in sorted(os.listdir(ori_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = ori_dir / fname
        out_path = out_dir / fname

        base = Path(fname).stem
        mask_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = mask_dir / f"{base}{ext}"
            if p.exists():
                mask_path = p
            break
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if img is None:
            print(f"이미지 읽기 실패: {fname}")
            continue

        if mask is None:
            print(f"마스크 없음 → 원본 복사: {fname}")
            cv2.imwrite(str(out_path), img)
            continue

        mask_bin = (mask > 127).astype(np.uint8)
        masked = img * mask_bin[:, :, None]
        cv2.imwrite(str(out_path), masked)
        #print(f"마스크 적용 완료: {fname}")

    print("모든 이미지 마스크 적용 완료 → 'images/' 폴더 저장됨")