import os
import cv2
import numpy as np
import matplotlib
import gradio as gr
try:
    import spaces  # Hugging Face Spaces ZeroGPU
except ImportError:
    # Chạy được trên máy local/Colab không cài spaces.
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn
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

# Quy tắc OpenCV cho biển chỉ dẫn nền xanh.
# Các ngưỡng có thể tinh chỉnh theo camera / độ phân giải.
CV_BLUE_HSV_LOW = (88, 85, 35)    # xanh dương (lọc nền xanh nhạt của nhà/sky)
CV_BLUE_HSV_HIGH = (135, 255, 255)
CV_STRONG_HSV_LOW = (90, 130, 40) # vùng xanh bão hòa dùng để tạo ứng viên
CV_ROI_BOTTOM = 0.85             # giới hạn tìm kiếm theo chiều cao ảnh
CV_MIN_ASPECT = 0.65             # bao gồm biển vuông và chữ nhật đứng
CV_MAX_ASPECT = 6.0              # cho phép bảng hướng dẫn dài ngang
CV_MIN_RECTANGULARITY = 0.56
CV_MIN_BLUE_RATIO = 0.43
CV_MIN_STRONG_RATIO = 0.27
CV_MIN_WHITE_RATIO = 0.008       # nội dung chữ/mũi tên trắng
CV_MIN_CORNER_BLUE = 0.22       # ít nhất 3/4 góc có nền xanh


def box_overlap_details(a, b):
    """(IoU, IoS, tỷ lệ của box b được phủ bởi box a)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, area_a + area_b - inter), inter / min(area_a, area_b), inter / area_b


def get_region_ratios(hsv, box):
    """Tỷ lệ xanh, xanh bão hòa và trắng trong vùng bounding box."""
    h, w = hsv.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0
    roi = hsv[y1:y2, x1:x2]
    area = float(roi.shape[0] * roi.shape[1])
    blue = cv2.inRange(roi, CV_BLUE_HSV_LOW, CV_BLUE_HSV_HIGH)
    strong = cv2.inRange(roi, CV_STRONG_HSV_LOW, CV_BLUE_HSV_HIGH)
    white = cv2.inRange(roi, (0, 0, 145), (179, 95, 255))
    return (cv2.countNonZero(blue) / area,
            cv2.countNonZero(strong) / area,
            cv2.countNonZero(white) / area)


def blue_rectangle_corners(hsv, box):
    """Kiểm tra bốn góc xanh để tránh nhầm biển tròn với biển chữ nhật."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    roi = hsv[y1:y2, x1:x2]
    hh, ww = roi.shape[:2]
    if hh < 8 or ww < 8:
        return 0
    blue = cv2.inRange(roi, CV_BLUE_HSV_LOW, CV_BLUE_HSV_HIGH) > 0
    dy, dx = max(2, int(hh * 0.14)), max(2, int(ww * 0.14))
    corners = (blue[:dy, :dx], blue[:dy, -dx:],
               blue[-dy:, :dx], blue[-dy:, -dx:])
    return sum(float(patch.mean()) >= CV_MIN_CORNER_BLUE for patch in corners)


def blue_guide_candidates(rgb):
    """Tìm biển nền xanh dạng vuông/chữ nhật bằng OpenCV, độc lập YOLO.

    Bước 1: HSV bão hòa cao tạo ứng viên, tránh tòa nhà/xe có màu xanh nhạt.
    Bước 2: morphology nhẹ để nối phần nền bị chia bởi ký tự trắng.
    Bước 3: lọc hình học + tỷ lệ xanh + chi tiết trắng; khử box trùng.

    `blue_ratio` KHÔNG phải xác suất/độ tin cậy của mô hình học máy.
    """
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return []

    roi_bottom = max(1, min(h, int(round(h * CV_ROI_BOTTOM))))
    hsv = cv2.cvtColor(rgb[:roi_bottom], cv2.COLOR_RGB2HSV)
    strong = cv2.inRange(hsv, CV_STRONG_HSV_LOW, CV_BLUE_HSV_HIGH)
    # Kernel nhỏ: kernel quá lớn sẽ nối biển xanh với tòa nhà/nhiều biển liền nhau.
    kw = 3 if w < 900 else 5
    kh = 3 if h < 650 else 5
    clean = cv2.morphologyEx(
        strong, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)),
    )
    # Không dilate mạnh vì biển đứng cạnh nhau cần giữ các box riêng.
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    min_w = max(18, int(round(w * 0.012)))
    min_h = max(12, int(round(h * 0.018)))
    min_area = max(150, int(round(w * h * 0.00035)))

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < min_w or bh < min_h:
            continue
        # Loại phần đuôi/nhánh xanh mảnh nối lên tòa nhà, cột hoặc đường dây.
        # Đặc biệt cần thiết với biển có ký hiệu trắng làm đứt nền xanh.
        raw_roi = strong[y:y + bh, x:x + bw]
        dense_rows = np.flatnonzero(np.mean(raw_roi > 0, axis=1) >= 0.23)
        dense_cols = np.flatnonzero(np.mean(raw_roi > 0, axis=0) >= 0.23)
        if len(dense_rows) == 0 or len(dense_cols) == 0:
            continue
        # Không trim quá sâu vào biển bị che một phần.
        trim_top = min(int(dense_rows[0]), int(bh * 0.30))
        trim_bottom = min(int(bh - 1 - dense_rows[-1]), int(bh * 0.30))
        trim_left = min(int(dense_cols[0]), int(bw * 0.20))
        trim_right = min(int(bw - 1 - dense_cols[-1]), int(bw * 0.20))
        x += trim_left
        y += trim_top
        bw -= trim_left + trim_right
        bh -= trim_top + trim_bottom
        area = bw * bh
        if bw < min_w or bh < min_h or area < min_area:
            continue
        # Chấm xanh rất nhỏ gần đáy ảnh/cạnh ảnh thường là xe hoặc vật thể nền.
        # Các biển ở xa phía trên vẫn được giữ lại.
        if (y + bh / 2) > h * 0.68 and bw < max(30, w * 0.06):
            continue
        aspect = bw / max(1, bh)
        if not CV_MIN_ASPECT <= aspect <= CV_MAX_ASPECT:
            continue
        # Loại dạng cột/nhánh xanh dài, không giống biển phẳng chữ nhật.
        rectangularity = min(1.0, cv2.contourArea(contour) / max(1.0, area))
        if rectangularity < CV_MIN_RECTANGULARITY:
            continue

        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        blue_ratio, strong_ratio, white_ratio = get_region_ratios(hsv, box)
        if blue_ratio < CV_MIN_BLUE_RATIO or strong_ratio < CV_MIN_STRONG_RATIO:
            continue
        # Biển chữ nhật phải giữ được nền xanh ở ít nhất 3 góc,
        # tránh gán nhầm các biển tròn bắt buộc (nền xanh).
        if blue_rectangle_corners(hsv, box) < 3:
            continue
        # Biển chỉ dẫn thường có chữ/mũi tên/ký hiệu trắng. Cho phép biển
        # rất xanh, chữ nhỏ hoặc mờ vẫn đi qua nếu hình dạng đủ rõ.
        if white_ratio < CV_MIN_WHITE_RATIO and not (strong_ratio >= 0.72 and rectangularity >= 0.80):
            continue
        # Tách hai bảng xanh liền sát nếu có đường phân cách dọc sắc nét
        # chạy gần hết chiều cao. Nếu không thấy seam, giữ nguyên bảng dài.
        splits = [box]
        if bw >= 2.1 * bh and bw >= 80:
            gray = cv2.cvtColor(rgb[y:y + bh, x:x + bw], cv2.COLOR_RGB2GRAY)
            gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            vertical_support = np.mean(gradient[2:-2] > 150, axis=0) if bh > 8 else np.zeros(bw)
            margin = max(16, int(bh * 0.35))
            seam_positions = np.where(vertical_support > 0.74)[0]
            seam_positions = [int(z) for z in seam_positions if margin <= z < bw - margin]
            if seam_positions:
                # Chỉ lấy đường có độ hỗ trợ lớn nhất, không cắt theo từng chữ.
                seam = max(seam_positions, key=lambda z: vertical_support[z])
                proposed = [np.array([x, y, x + seam, y + bh], dtype=np.float32),
                            np.array([x + seam + 1, y, x + bw, y + bh], dtype=np.float32)]
                if all((p[2] - p[0] >= min_w) and
                       (get_region_ratios(hsv, p)[0] >= CV_MIN_BLUE_RATIO) and
                       (blue_rectangle_corners(hsv, p) >= 3)
                       for p in proposed):
                    splits = proposed

        for sign_box in splits:
            local_blue, local_strong, local_white = get_region_ratios(hsv, sign_box)
            candidates.append({
                "box": sign_box, "blue_ratio": float(local_blue),
                "strong_ratio": float(local_strong),
                "white_ratio": float(local_white),
                "rectangularity": float(rectangularity),
            })

    # Tránh một biển được thêm hai lần khi các vùng chồng lấn.
    candidates.sort(key=lambda c: c["strong_ratio"] *
                    (c["box"][2] - c["box"][0]) *
                    (c["box"][3] - c["box"][1]), reverse=True)
    result = []
    for item in candidates:
        if not any(box_overlap_details(item["box"], other["box"])[1] >= 0.70
                   for other in result):
            result.append(item)
    return result

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
    cv_candidates = []
    if use_cv:
        # Detect từ OpenCV KHÔNG phụ thuộc việc YOLO có thấy biển xanh hay không.
        cv_candidates = blue_guide_candidates(rgb)
        hsv_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        matched_yolo_indices = set()
        for candidate in cv_candidates:
            guide_box = candidate["box"]
            matches = []
            for j, item in enumerate(items):
                iou, ios, cover_yolo = box_overlap_details(guide_box, item["box"])
                # Không ghi đè biển cấm màu đỏ chỉ vì box nằm gần biển xanh.
                blue_in_yolo, _, _ = get_region_ratios(hsv_full, item["box"])
                if blue_in_yolo < 0.35:
                    continue
                if iou >= 0.35 or (cover_yolo >= 0.68 and ios >= 0.68):
                    matches.append((j, iou, cover_yolo))

            if matches:
                # Nhiều box YOLO cùng nằm trong một cụm bảng xanh:
                # giữ từng box, sửa nhãn, không dùng 1 box lớn phủ tất cả.
                for j, _, _ in matches:
                    if j in matched_yolo_indices:
                        continue
                    items[j].update({
                        "name": "Biển chỉ dẫn",
                        "source": "YOLO + OpenCV",
                        "blue_ratio": candidate["blue_ratio"],
                    })
                    matched_yolo_indices.add(j)
                    corrected += 1
            else:
                # YOLO bỏ sót: OpenCV thêm kết quả mới riêng biệt.
                # Tránh thêm lại phần lớn một biển đã có box YOLO ở cùng vị trí.
                if any(box_overlap_details(guide_box, item["box"])[0] >= 0.55
                       for item in items):
                    continue
                items.append({
                    "box": guide_box,
                    "name": "Biển chỉ dẫn",
                    "yolo_conf": None,
                    "source": "OpenCV",
                    "blue_ratio": candidate["blue_ratio"],
                })
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
        except Exception as exc:
            print(f"SAM segmentation error: {type(exc).__name__}: {exc}")

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

        if item["yolo_conf"] is not None:
            score_label = f"YOLO: {item['yolo_conf'] * 100:.1f}% | {item['source']}"
            bar_value = item["yolo_conf"]
        else:
            score_label = f"OpenCV: vùng xanh {item['blue_ratio'] * 100:.1f}%"
            bar_value = item["blue_ratio"]  # tỷ lệ pixel xanh, KHÔNG phải confidence

        d.text((tx, y + 96), score_label, font=font_info, fill=(50, 60, 75))
        bw = min(295, max_txt)
        d.rounded_rectangle((tx, y + 147, tx + bw, y + 155), radius=4, fill=(230, 234, 240))
        d.rounded_rectangle((tx, y + 147, tx + max(1, bw * bar_value), y + 155), radius=4, fill=color)

    info_text = (f"Phát hiện tổng cộng: {n} biển báo\n"
                 f"YOLO: {len(res.boxes)} | OpenCV bổ sung: {added} | "
                 f"OpenCV hiệu chỉnh: {corrected}\n"
                 f"Ứng viên biển xanh qua lọc hình học/màu: {len(cv_candidates)}"
                 if use_cv else f"Phát hiện {n} biển báo từ YOLO (OpenCV tắt).")
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
        gr.Textbox(label="Thống kê"),
    ],
    title="🚦 Nhận diện & Phân đoạn Biển báo Giao thông VN",
    description="Hệ thống tích hợp YOLOv12m + SAM + OpenCV chạy trên Hugging Face Spaces (ZeroGPU)."
)

if __name__ == "__main__":
    demo.launch()
