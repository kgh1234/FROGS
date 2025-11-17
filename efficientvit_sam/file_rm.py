import os

def clean_mask_folders(root_path):
    """
    /root/scan*/mask 폴더 내에서
    '_max_mask.png' 파일만 남기고 나머지 모두 삭제
    """
    for scan_name in sorted(os.listdir(root_path)):
        scan_path = os.path.join(root_path, scan_name)
        mask_dir = os.path.join(scan_path, "mask")

        if not os.path.isdir(mask_dir):
            continue

        files = os.listdir(mask_dir)
        removed = 0
        kept = 0

        for f in files:
            fpath = os.path.join(mask_dir, f)

            # '_max_mask.png'만 유지
            if f.endswith("_max_mask.png"):
                kept += 1
                continue

            # 나머지는 삭제
            try:
                os.remove(fpath)
                removed += 1
            except Exception as e:
                print(f"⚠️ 삭제 실패: {fpath} ({e})")

        print(f"🧹 {scan_name}/mask 정리 완료 — 남김: {kept}, 삭제: {removed}")

    print("\n✅ 모든 mask 폴더 정리 완료!")


# 사용 예시
if __name__ == "__main__":
    ROOT_PATH = "../../masked_datasets/DTU_chaewon"   # 👉 여기에 상위 폴더 경로 지정
    clean_mask_folders(ROOT_PATH)
