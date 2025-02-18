import cv2
import glob
import os
import yaml
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO("runs/detect/train5/weights/best.pt")

# สร้างโฟลเดอร์ที่ใช้บันทึกผลลัพธ์
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# โหลดไฟล์ data.yaml เพื่อดึงชื่อคลาส
with open("dataset/data.yaml", "r") as file:
    data = yaml.safe_load(file)
    class_names = data['names']  # ดึงชื่อคลาสจาก data.yaml

# ดึงลิสต์ไฟล์ทั้งหมดจากโฟลเดอร์
image_paths = glob.glob("img/*.jpg")  # หรือ *.png ตามประเภทไฟล์

# กำหนดสีของคลาส
custom_colors = {
    0: (255, 0, 0),   # Class 0 -> แดง
    1: (0, 255, 0),   # Class 1 -> เขียว
    2: (0, 0, 255),   # Class 2 -> น้ำเงิน
    3: (255, 255, 0), # Class 3 -> เหลือง
    # เพิ่มสีสำหรับคลาสอื่นๆ ที่มีใน dataset
}

# วนลูปอ่านภาพทีละไฟล์
for img_path in image_paths:
    img = cv2.imread(img_path)

    # ทำนายผล
    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # ดึง class_id

            # ใช้สีที่กำหนดตาม class_id
            color = custom_colors.get(class_id, (255, 255, 255))  # สีขาวถ้าไม่เจอคลาส

            # วาดกรอบ
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # ใส่ชื่อคลาสที่ดึงมาจาก data.yaml
            class_name = class_names[class_id]  # ดึงชื่อคลาสจาก data.yaml
            cv2.putText(img, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # บันทึกผลลัพธ์
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, img)

print("ผลลัพธ์ถูกบันทึกในโฟลเดอร์:", output_folder)
