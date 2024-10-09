import cv2
import numpy as np
import easyocr
from ultralytics import YOLOv10
import csv
import datetime
import os

# Khởi tạo EasyOCR reader với tùy chỉnh nhiều ngôn ngữ nếu cần
reader = easyocr.Reader(['en'], gpu=False)

# Khởi tạo mô hình YOLO
model = YOLOv10("weights/best_licenseplates.pt")

# Tạo thư mục để lưu ảnh biển số nếu chưa tồn tại
if not os.path.exists('saved_plates'):
    os.makedirs('saved_plates')

# Hàm tiền xử lý hình ảnh biển số
def preprocess_plate_image(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Giảm kích thước kernel để giữ chi tiết
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Tăng clipLimit để tăng độ tương phản
    enhanced = clahe.apply(blurred)
    _, binary_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img

# Hàm xử lý hình ảnh và nhận diện biển số
def process_image(image):
    img_array = np.array(image)
    results = model(img_array)

    # Kiểm tra xem có boxes trong kết quả hay không
    if not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
        return img_array, [], False, None

    boxes = results[0].boxes
    probs = results[0].probs if hasattr(results[0], 'probs') else None

    # Nếu không có boxes hoặc probs, hoặc không có boxes được phát hiện
    if boxes is None or len(boxes) == 0 or (probs is not None and len(probs) == 0):
        return img_array, [], False, None

    # Tìm kết quả có độ tin cậy cao nhất
    max_conf_index = np.argmax(probs[:, -1]) if probs is not None else 0

    box = boxes[max_conf_index]
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Truy xuất các giá trị tọa độ
    conf = float(probs[max_conf_index, -1]) if probs is not None else 1.0  # Truy xuất giá trị độ tin cậy
    cls = int(box.cls[0]) if hasattr(box, 'cls') else 0  # Truy xuất giá trị lớp

    top_left = (x1, y1)
    bottom_right = (x2, y2)
    plate_img = img_array[y1:y2, x1:x2]
    plate_img = preprocess_plate_image(plate_img)
    result = reader.readtext(plate_img, paragraph="True", min_size=120, text_threshold=0.8, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    recognized_texts = []
    for detection in result:
        text = detection[1]
        recognized_texts.append(text)

    font = cv2.FONT_HERSHEY_COMPLEX
    img_array = cv2.rectangle(img_array, top_left, bottom_right, (255, 255, 0), 2)
    img_array = cv2.putText(img_array, ''.join(recognized_texts).replace(" ", ""), (top_left[0], top_left[1] - 10), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    plate_text = ''.join(recognized_texts).replace(" ", "")
    status = check_plate_status(plate_text)

    # Chỉ lưu ảnh biển số xe nếu trạng thái là "vào"
    if status == 'vào':
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plate_filename = f'saved_plates/{plate_text}_{current_time}.png'
        cv2.imwrite(plate_filename, img_array[y1:y2, x1:x2])
    elif status == 'ra':
        # Xóa ảnh biển số xe khi trạng thái là "ra"
        for file_name in os.listdir('saved_plates'):
            if plate_text in file_name:
                os.remove(os.path.join('saved_plates', file_name))

    return img_array, recognized_texts, True, status

# Hàm kiểm tra trạng thái xe vào/ra
def check_plate_status(plate_text):
    with open('number_plate.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    last_entry = None
    for row in rows:
        if row[1] == plate_text:
            last_entry = row

    if last_entry:
        if last_entry[2] == 'vào':
            return 'ra'
        else:
            return 'vào'
    else:
        return 'vào'

# Hàm kiểm tra và cập nhật trạng thái xe vào/ra trong file CSV
def update_csv_file(plate_texts, status):
    with open('number_plate.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    plate_text = ''.join(plate_texts).replace(" ", "")
    new_rows = []
    found_plate = False

    for row in rows:
        if row[1] == plate_text:
            found_plate = True
            if status == 'ra':
                print(f"Xe biển số {plate_text} đã đi ra.")
            else:
                print(f"Xe biển số {plate_text} đã đi vào.")
        else:
            new_rows.append(row)

    if not found_plate or status == 'vào':
        current_time = datetime.datetime.now()
        new_rows.append([current_time.strftime("%d-%m-%Y %H:%M:%S"), plate_text, status])
        print(f"Biển số xe nhận diện được là: {plate_text} ({status})")

    with open('number_plate.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
