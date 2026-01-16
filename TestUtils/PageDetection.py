import torch
import pathlib
import os
import cv2
import shutil
from collections import defaultdict
from pathlib import Path

pathlib.WindowsPath = pathlib.PosixPath
save_dir = './runs/detect/exp/crops'
model = torch.hub.load('./yolov5', 'custom', path= './best.pt', source='local')
model.conf = 0.5
m_center = None

def save_crop_half(xyxy, im, m_center, file_path, vertical=True, BGR=True):
    """
    Saves two halves of the cropped image based on m_center.

    Parameters:
    xyxy (tuple): The bounding box coordinates (x_min, y_min, x_max, y_max).
    im (ndarray): The image from which to crop.
    m_center (tuple): The center point (x, y) to divide the crop.
    file_path (Path): The base path to save the cropped images.
    vertical (bool): If True, split vertically based on m_center, else horizontally.
    BGR (bool): If True, save images in BGR format, else in RGB.
    """
    # Crop the original bounding box
    crop = im[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

    # Calculate the split point relative to the crop
    split_point = int(m_center[1] - xyxy[1]) if vertical else int(m_center[0] - xyxy[0])

    # Split the crop into two halves
    if vertical:
        top_half = crop[:split_point, :]
        bottom_half = crop[split_point:, :]
    else:
        left_half = crop[:, :split_point]
        right_half = crop[:, split_point:]

    # Save each half
    for idx, half in enumerate([top_half, bottom_half] if vertical else [left_half, right_half], 1):
        print(idx)
        if idx ==1:
            folder_path = os.path.join(file_path, "Left")
            os.makedirs(folder_path, exist_ok=True)
            half_path = os.path.join(folder_path, "page_capture.jpg")
        if idx ==2:
            folder_path = os.path.join(file_path, "Right")
            os.makedirs(folder_path, exist_ok=True)
            half_path = os.path.join(folder_path, "page_capture.jpg")
        if BGR:
            cv2.imwrite(str(half_path), half)
        else:
            half = cv2.cvtColor(half, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(half_path), half)

def find_center(box):
    """
    Function to find the center coordinates based on a single bounding box coordinates

    Parameters:
    box (list of tensors): List of tensors representing a bounding box [x_min, y_min, x_max, y_max]

    Returns:
    tuple: A tuple representing the center coordinates (x_center, y_center) of the box
    """
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center.item(), y_center.item())

print('start')
image_path = "./page_capture.jpg"
out = model(image_path)
print(out)

#crop save
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
img = cv2.imread(image_path)
class_index = defaultdict(int)

sorted_preds = sorted(out.pred[0], key=lambda x: x[4], reverse=False)
for i, pred in enumerate(sorted_preds):
    if pred is not None:
        xyxy = pred[:4].int()
        conf = pred[4]
        cls = int(pred[5])
        class_name = out.names[cls]
        x1, y1, x2, y2 = xyxy
        print(f"Object {i}: {class_name} - Coordinates: ({x1}, {y1}, {x2}, {y2}) - Confidence: {conf:.2f}")
        # 클래스별 폴더 생성
        class_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # 객체 크롭 및 저장
        cropped_img = img[y1:y2, x1:x2]
        save_path = os.path.join(class_dir, f"page_capture{class_index[class_name]}.jpg")
       
        cv2.imwrite(save_path, cropped_img)
        print(f"Object {i} saved: {save_path}")
        class_index[class_name] += 1

        if class_name == 'm':
            m_center = find_center(xyxy)
        if class_name == 'b':
            if m_center is None:
                m_center = find_center(xyxy)
            save_crop_half(xyxy, img, m_center, save_dir, vertical=False, BGR=True)

out.save()

