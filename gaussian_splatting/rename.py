import os, re, shutil, glob
from PIL import Image

IMG_DIR = "/home/knuvi/Desktop/gahyeon/dataset/garden/images"
MASK_DIR = "/home/knuvi/Desktop/gahyeon/dataset/garden/table_mask"
OUT_DIR  = os.path.join(os.path.dirname(MASK_DIR), "table_mask_renamed")
os.makedirs(OUT_DIR, exist_ok=True)

# 마스크 이름 패턴: <stem>_...mask....png  (대소문자 무시)
mask_suffix_re = re.compile(r"^(?P<stem>.+?)_.*mask.*\.(png|jpg|jpeg)$", re.IGNORECASE)

# 이미지 목록(확장자 혼용 대응)
img_paths = []
for ext in ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG"):
    img_paths += glob.glob(os.path.join(IMG_DIR, ext))

found, missing = 0, 0
for ip in sorted(img_paths):
    stem = os.path.splitext(os.path.basename(ip))[0]  # e.g., DSC07956
    # 해당 스템으로 시작하고 'mask'를 포함하는 마스크 검색
    candidates = []
    for ext in ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"):
        candidates += glob.glob(os.path.join(MASK_DIR, f"{stem}*{ext}"))
    # 필터: 이름에 'mask' 포함
    candidates = [p for p in candidates if re.search(r"mask", os.path.basename(p), re.IGNORECASE)]
    if not candidates:
        missing += 1
        print(f"[WARN] 마스크 없음: {stem}")
        continue
    if len(candidates) > 1:
        # 가장 짧은 파일명을 우선(가끔 *_bike_mask, *_mask 등 여러개 있을 때)
        candidates.sort(key=lambda p: len(os.path.basename(p)))
    mp = candidates[0]

    # 흑백 0/255 보장 및 PNG로 저장
    try:
        im = Image.open(mp).convert("L")
        # 이진화(이미 이진이면 변화 없음)
        im = im.point(lambda x: 255 if x >= 128 else 0, mode="L")
        out_path = os.path.join(OUT_DIR, f"{stem}.png")   # 최종 이름: DSC07956.png
        im.save(out_path)
        found += 1
    except Exception as e:
        print(f"[ERR] {mp} 처리 실패: {e}")

print(f"완료. 매칭 {found}개, 누락 {missing}개. 출력 폴더: {OUT_DIR}")
