import os
import cv2
import numpy as np
import matplotlib
import gradio as gr
import spaces  # Bắt buộc phải có để mở khóa ZeroGPU
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO, SAM

MODEL_PATH = "best.pt"
SAM_PATH = "sam_b.pt"
MIN_WIDTH = 1000
RIGHT_WIDTH = 640
CROP_SIZE = 140
ROW_HEIGHT = 180
PAD = 22
HEADER_H = 76
NAME_PREFIX = "Biển "

PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128),
]

FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
F_REG = os.path.join(FONT_DIR, "DejaVuSans.ttf")
F_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

def name_vi(name):
    name = str(name).replace("_", " ").strip()
    return name if name.lower().startswith("biển") else NAME_PREFIX + name

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1, aa + ab - inter), inter / max(1, min(aa, ab))

def blue_guide_candidates(rgb):
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, (90, 65, 35), (135, 255, 255))
    k = max(3, int(min(h, w) * 0.012) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if (bw < max(25, w * 0.025) or bh < max(25, h * 0.025) or area < max(700, 0.0008 * w * h)):
            continue
        aspect = bw / bh
        if not (0.65 <= aspect <= 6.0):
            continue
        rectangularity = cv2.contourArea(cnt) / max(1, area)
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if rectangularity < 0.77 or len(approx) not in (4, 5):
            continue
        blue_ratio = cv2.countNonZero(blue[y:y+bh, x:x+bw]) / area
        if blue_ratio < 0.52:
            continue
        m = max(2, int(min(bw, bh) * 0.09))
        inner = hsv[y+m:y+bh-m, x+m:x+bw-m]
        if inner.size == 0:
            continue
        white_ratio = np.mean((inner[:, :, 1] < 115) & (inner[:, :, 2] > 120))
        if white_ratio < 0.012:
            continue
        candidates.append({"box": np.array([x, y, x+bw, y+bh], dtype=float), "blue_ratio": float(blue_ratio)})
    candidates.sort(key=lambda z: (z["box"][2] - z["box"][0]) * (z["box"][3] - z["box"][1]), reverse=True)
    distinct = []
    for candidate in candidates:
        if not any(box_iou(candidate["box"], old["box"])[0] > 0.55 for old in distinct):
            distinct.append(candidate)
    return distinct

def crop_original(rgb, box, padding=0.10):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return Image.fromarray(rgb[y1:y2, x1:x2]) if x2 > x1 and y2 > y1 else None

def wrap_lines(draw, text, font, max_width, limit=2):
    lines, line = [], ""
    for word in text.split():
        test = (line + " " + word).strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    if len(lines) > limit:
        lines = lines[:limit]
        last = lines[-1]
        while last and draw.textbbox((0, 0), last + "...", font=font)[2] > max_width:
            last = last[:-1]
        lines[-1] = last + "..."
    return lines

# Khởi tạo mô hình
det_model = YOLO(MODEL_PATH)
sam_model = SAM(SAM_PATH)

@spaces.GPU  # Định thị yêu cầu cấp phát GPU khi người dùng chạy hàm nhận diện này
def predict(image, conf_thres, use_sam, use_cv):
    if image is None:
        return None, "Vui lòng tải lên một ảnh!"
    
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = rgb.shape[:2]

    res = det_model.predict(source=bgr, conf=conf_thres, verbose=False)[0]
    items = []
    for box, cls_id, score in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.cls.cpu().numpy().astype(int),
        res.boxes.conf.cpu().numpy()
    ):
        items.append({
            "box": box.astype(float),
            "name": name_vi(det_model.names[int(cls_id)]),
            "yolo_conf": float(score),
            "source": "YOLO",
            "blue_ratio": None,
        })

    added, corrected = 0, 0
    if use_cv:
        blue_boxes = blue_guide_candidates(rgb)
        for candidate in blue_boxes:
            guide_box = candidate["box"]
            match, best_overlap = None, 0
            for j, item in enumerate(items):
                iou, coverage = box_iou(guide_box, item["box"])
                overlap = max(iou, coverage)
                if overlap > best_overlap and (iou >= 0.25 or coverage >= 0.70):
                    match, best_overlap = j, overlap
            if match is not None:
                items[match].update({"box": guide_box, "name": "Biển chỉ dẫn", "source": "YOLO + OpenCV", "blue_ratio": candidate["blue_ratio"]})
                corrected += 1
            else:
                items.append({"box": guide_box, "name": "Biển chỉ dẫn", "yolo_conf": None, "source": "OpenCV", "blue_ratio": candidate["blue_ratio"]})
                added += 1

    items.sort(key=lambda z: (z["box"][0] + z["box"][2]) / 2)

    canvas = rgb.astype(np.float32)
    mask_contours = []
    if use_sam and items:
        try:
            sam_result = sam_model(bgr, bboxes=[x["box"].tolist() for x in items], verbose=False)[0]
            masks = sam_result.masks.data.cpu().numpy() if sam_result.masks is not None else []
            for i, mask in enumerate(masks[:len(items)]):
                if mask.shape != (H, W):
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
                color = np.array(PALETTE[i % len(PALETTE)])
                canvas[mask] = 0.55 * canvas[mask] + 0.45 * color
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_contours.append((cnts, tuple(map(int, color))))
        except Exception:
            pass

    annotated = canvas.astype(np.uint8)
    for cnts, color in mask_contours:
        cv2.drawContours(annotated, cnts, -1, color, 2)

    scale = max(1.0, MIN_WIDTH / W)
    if scale > 1:
        annotated = cv2.resize(annotated, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_CUBIC)
    Wn, Hn = annotated.shape[1], annotated.shape[0]
    left = Image.fromarray(annotated)
    d = ImageDraw.Draw(left)
    f_num = ImageFont.truetype(F_BOLD, 21)

    for i, item in enumerate(items):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = item["box"] * scale
        d.rectangle((x1, y1, x2, y2), outline=color, width=3)
        r = 18
        cx = min(max(x1 + r * 0.2, r + 2), Wn - r - 2)
        cy = max(r + 2, y1 - r)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline="white", width=2)
        d.text((cx, cy), str(i + 1), fill="white", font=f_num, anchor="mm")

    n = len(items)
    out_w = Wn + RIGHT_WIDTH
    out_h = max(Hn + HEADER_H, HEADER_H + PAD + max(1, n) * ROW_HEIGHT + PAD)
    out = Image.new("RGB", (out_w, out_h), (247, 249, 252))
    d = ImageDraw.Draw(out)

    font_title = ImageFont.truetype(F_BOLD, 23)
    font_name = ImageFont.truetype(F_BOLD, 19)
    font_info = ImageFont.truetype(F_REG, 16)
    font_small = ImageFont.truetype(F_REG, 15)

    d.rectangle((0, 0, Wn, HEADER_H), fill=(238, 242, 248))
    d.text((PAD, 15), "YOLO12m + SAM + OpenCV | Detection & Segmentation", font=font_title, fill=(25, 40, 65))
    d.text((PAD, 45), f"Phát hiện {n} biển báo giao thông", font=font_small, fill=(95, 105, 120))
    out.paste(left, (0, HEADER_H))

    d = ImageDraw.Draw(out)
    d.line(((Wn, 0), (Wn, out_h)), fill=(210, 215, 225), width=2)
    d.text((Wn + PAD, 15), "KẾT QUẢ NHẬN DIỆN", font=font_title, fill=(25, 40, 65))
    d.text((Wn + PAD, 45), f"{n} biển được tìm thấy", font=font_small, fill=(95, 105, 120))

    for i, item in enumerate(items):
        color = PALETTE[i % len(PALETTE)]
        x, y = Wn + PAD, HEADER_H + PAD + i * ROW_HEIGHT
        cw, ch = RIGHT_WIDTH - 2 * PAD, ROW_HEIGHT - 12
        d.rounded_rectangle((x, y, x + cw, y + ch), radius=12, fill="white", outline=(220, 225, 232))
        d.rounded_rectangle((x, y + 12, x + 5, y + ch - 12), radius=2, fill=color)

        crop = crop_original(rgb, item["box"])
        px, py = x + 15, y + (ch - CROP_SIZE) // 2
        d.rounded_rectangle((px, py, px + CROP_SIZE, py + CROP_SIZE), radius=8, fill=(242, 244, 247))
        if crop is not None:
            thumb = ImageOps.contain(crop, (CROP_SIZE - 8, CROP_SIZE - 8), method=Image.Resampling.LANCZOS)
            out.paste(thumb, (px + (CROP_SIZE - thumb.width) // 2, py + (CROP_SIZE - thumb.height) // 2))

        tx = px + CROP_SIZE + 22
        max_txt = x + cw - tx - 12
        d.text((tx, y + 14), f"BIỂN BÁO #{i + 1}", font=font_small, fill=color)
        for j, line in enumerate(wrap_lines(d, item["name"], font_name, max_txt)):
            d.text((tx, y + 37 + j * 25), line, font=font_name, fill=(30, 35, 45))

        if item["source"] == "YOLO":
            score_label = f"Độ tin cậy: {item['yolo_conf'] * 100:.1f}%"
            bar_value = item["yolo_conf"]
        else:
            score_label = f"Biển chỉ dẫn (OpenCV)"
            bar_value = item["blue_ratio"]

        d.text((tx, y + 96), score_label, font=font_info, fill=(50, 60, 75))
        bw = min(295, max_txt)
        d.rounded_rectangle((tx, y + 147, tx + bw, y + 155), radius=4, fill=(230, 234, 240))
        d.rounded_rectangle((tx, y + 147, tx + max(1, bw * bar_value), y + 155), radius=4, fill=color)

    info_text = f"✅ Phát hiện tổng cộng {n} biển báo."
    return out, info_text

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Tải ảnh lên"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Ngưỡng Confidence YOLO"),
        gr.Checkbox(value=True, label="Bật SAM Segmentation"),
        gr.Checkbox(value=True, label="Bật OpenCV biển xanh"),
    ],
    outputs=[
        gr.Image(type="pil", label="Kết quả"),
        gr.Textbox(label="Thông kê"),
    ],
    title="🚦 Nhận diện & Phân đoạn Biển báo Giao thông VN",
    description="Hệ thống tích hợp YOLOv12m + SAM + OpenCV chạy trên Hugging Face Spaces (ZeroGPU)."
)

if __name__ == "__main__":
    demo.launch()