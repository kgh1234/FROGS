import torch, cv2
from groundingdino.util.inference import load_model, load_image, predict
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

# --- GroundingDINO: text -> boxes ---
GDINO_CFG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WTS = "weights/groundingdino_swin_t_ogc.pth"   # 공개 체크포인트 사용
TEXT = "a table"  # 여기에 텍스트 프롬프트

gdino = load_model(GDINO_CFG, GDINO_WTS)
image_src, image = load_image("/home/knuvi/Desktop/gahyeon/dataset/garden/images_ori/DSC07956.JPG")
boxes, logits, phrases = predict(
    model=gdino, image=image, caption=TEXT,
    box_threshold=0.25, text_threshold=0.25
)

# boxes: [x1,y1,x2,y2] (xyxy, 이미지 정규화 좌표일 수 있음 → 픽셀로 변환)
h, w = image_src.shape[:2]
boxes_px = []
for (x1,y1,x2,y2) in boxes:
    boxes_px.append([int(x1*w), int(y1*h), int(x2*w), int(y2*h)])

# --- EfficientViT-SAM: box -> mask ---
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = create_efficientvit_sam_model(name="efficientvit-sam-l0", pretrained=False)
sam.load_state_dict(torch.load("efficientvit_sam/weight/efficientvit_sam_l0.pt", map_location=device))
sam = sam.to(device).eval()
predictor = EfficientViTSamPredictor(sam)

img_bgr = cv2.imread("/home/knuvi/Desktop/gahyeon/dataset/garden/images_ori/DSC07956.JPG")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
predictor.set_image(img_rgb)

import numpy as np
all_masks = []
for bx in boxes_px:
    # SAM은 xyxy 박스 프롬프트
    masks, scores, _ = predictor.predict(box=np.array(bx))
    # 메인 마스크만 취함
    best = masks[np.argmax(scores)]
    all_masks.append(best)

# 시각화 (간단히)
vis = img_bgr.copy()
for m in all_masks:
    vis[m>0] = (0.3*vis[m>0] + 0.7*[0,255,0]).astype(vis.dtype)
cv2.imwrite("outputs/text_prompt_result.png", vis)
print("saved -> outputs/text_prompt_result.png")
