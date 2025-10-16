import os
import time
import numpy as np
import torch
import cv2
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor


CONFIG_FILE = "/workspace/gahyeon/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "/workspace/gahyeon/GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAM_CKPT = "/workspace/gahyeon/efficientvit_sam/weight/efficientvit_sam_xl1.pt"

IMAGE_DIR = "/workspace/gahyeon/dataset/bicycle/images"
OUTPUT_DIR = "/workspace/gahyeon/dataset/bicycle-mask/bicycle_bonsai_box0.65_text0.4"

TEXT_PROMPT = "a flower, a bonsai tree, a pot, a leaf, a tree"
BOX_THRESHOLD = 0.65
TEXT_THRESHOLD = 0.4
CPU_ONLY = False



def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("GroundingDINO load:", load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, cpu_only=False):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask].cpu()
    boxes_filt = boxes[filt_mask].cpu()

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(phrase)

    return boxes_filt, pred_phrases


def main():
    device = "cuda" if torch.cuda.is_available() and not CPU_ONLY else "cpu"

    dino_model = load_model(CONFIG_FILE, CHECKPOINT_PATH, cpu_only=CPU_ONLY)
    sam = create_efficientvit_sam_model("efficientvit-sam-xl1", pretrained=False)
    sam.load_state_dict(torch.load(SAM_CKPT, map_location=device))
    sam = sam.to(device).eval()
    predictor = EfficientViTSamPredictor(sam)

    #black_count = 0
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    total_time, count = 0, 0
    for fname in img_files:
        img_path = os.path.join(IMAGE_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(fname)[0]}_mask.png")
        print(f"🔹 Processing {img_path}")

        start = time.time()
        image_pil, image_tensor = load_image(img_path)

        boxes, phrases = get_grounding_output(
            dino_model, image_tensor, TEXT_PROMPT,
            BOX_THRESHOLD, TEXT_THRESHOLD, cpu_only=CPU_ONLY
        )

        image_np = np.array(image_pil)

        if boxes.shape[0] == 0:
            # fallback: 검정 마스크 저장
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            cv2.imwrite(out_path, mask)
            print(f"⚠️ No detection, saved black mask at {out_path}")
            continue

        predictor.set_image(image_np)

        for i, box in enumerate(boxes):
            h, w = image_pil.size[1], image_pil.size[0]
            box = box * torch.Tensor([w, h, w, h])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            box = box.cpu().numpy().astype(int)

            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255
            cv2.imwrite(out_path, mask)

            print(f"✅ Saved {out_path} (box={box}, phrase={phrases[i]})")
            break

        elapsed = time.time() - start
        total_time += elapsed
        count += 1

    if count > 0:
        print(f"\n📊 Average time per image: {total_time / count:.2f}s over {count} images")


if __name__ == "__main__":
    main()
