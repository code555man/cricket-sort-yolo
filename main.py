from ultralytics import YOLO
import cv2
import glob

# โหลดโมเดล YOLOv10 (ใช้เวอร์ชันเล็กสุด)
model = YOLO("model\yolov8n.pt")

# # เทรนโมเดล
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)

# โหลดโมเดลที่เทรนเสร็จ
model = YOLO("runs/detect/train/weights/best.pt")

# กำหนดสีแต่ละคลาส
custom_colors = {
    0: (255, 0, 0),  # แดง
    1: (0, 255, 0),  # เขียว
    2: (0, 0, 255)   # น้ำเงิน
}

# ทดสอบกับภาพใหม่
img_path = glob.glob("img/*.jpg")
results = model(img_path, save=True)

# แสดงผลลัพธ์
for result in results:
    print(result.boxes)
