import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
import uuid
import json
import time
import cv2
import requests
import os
import io

def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)

    if platform.system() == 'Darwin':
        font = 'AppleGothic.ttf'
    elif platform.system() == 'Windows':
        font = 'malgun.ttf'

    image_font = ImageFont.truetype(font, font_size)
    draw = ImageDraw.Draw(image)

    draw.text((x, y), text, font=image_font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

def Api_ocr_image(api_url, secret_key, image_path):
    files = [('file', open(image_path, 'rb'))]

    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    headers = {
        'X-OCR-SECRET': secret_key,
    }

    response = requests.post(api_url, headers=headers, data=payload, files=files)
    result = response.json()
 
    recognized_text = ""

    for field in result['images'][0]['fields']:
        text = field['inferText']
        recognized_text += text + " "
    
    return recognized_text.strip()   
       
def Api_ocr_frame(api_url, secret_key, frame):
    # 메모리 내에서 이미지를 바이트로 변환
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        print("프레임을 바이트로 변환하는데 실패했습니다.")
        return ""
    
    io_buf = io.BytesIO(buffer)

    files = [('file', io_buf)]

    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    headers = {
        'X-OCR-SECRET': secret_key,
    }

    response = requests.post(api_url, headers=headers, data=payload, files=files)
    result = response.json()

    recognized_text = ""

    for field in result['images'][0]['fields']:
        text = field['inferText']
        recognized_text += text + " "

    return recognized_text.strip()

def Api_ocr_process_image_with_page_number(api_url, secret_key, image_path):
    files = [('file', open(image_path, 'rb'))]

    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    headers = {
        'X-OCR-SECRET': secret_key,
    }

    response = requests.post(api_url, headers=headers, data=payload, files=files)
    result = response.json()

    recognized_text = ""
    removed_number = None

    for field in result['images'][0]['fields']:
        text = field['inferText']
       
        # Check if the first word is a number 
        # 첫글자가 숫자이면 페이지 수로 인식
        if removed_number is None and text.strip().split()[0].isdigit():
            removed_number = int(text.strip().split()[0])
            continue

        # Check if the last word is a number
        if text.strip().split()[-1].isdigit():
            removed_number = int(text.strip().split()[-1])
            break

        recognized_text += text + " "     

    if removed_number is not None:
        # Remove the found number from recognized_text 
        # 내용 중에 페이지 수와 같은 숫자가 있으면 삭제되는 문제 있음
        recognized_text = recognized_text.replace(str(removed_number), '', 1)
        
    return recognized_text.strip(), removed_number

def Api_ocr_pagenumber(api_url, secret_key, frame):
    # 메모리 내에서 이미지를 바이트로 변환
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        print("프레임을 바이트로 변환하는데 실패했습니다.")
        return ""
    
    io_buf = io.BytesIO(buffer)

    files = [('file', io_buf)]

    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    headers = {
        'X-OCR-SECRET': secret_key,
    }

    response = requests.post(api_url, headers=headers, data=payload, files=files)
    result = response.json()

    roi_img = frame
    
    recognized_text = ""
    removed_number = None

    for field in result['images'][0]['fields']:
        text = field['inferText']
       
        # Check if the first word is a number
        if removed_number is None and text.strip().split()[0].isdigit():
            removed_number = int(text.strip().split()[0])
            continue

        # Check if the last word is a number
        if text.strip().split()[-1].isdigit():
            removed_number = int(text.strip().split()[-1])
            break

        recognized_text += text + " "
       
        vertices_list = field['boundingPoly']['vertices']
        pts = [tuple(vertice.values()) for vertice in vertices_list]
        topLeft = [int(_) for _ in pts[0]]
        topRight = [int(_) for _ in pts[1]]
        bottomRight = [int(_) for _ in pts[2]]
        bottomLeft = [int(_) for _ in pts[3]]

        cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)

    if removed_number is not None:
        # Remove the found number from recognized_text
        recognized_text = recognized_text.replace(str(removed_number), '', 1)

    file_path = 'C:/Users/hyemi/Desktop/number.txt'  # 페이지 저장할 장소
    with open(file_path, 'r') as file:
        existing_number = int(file.read())

    if removed_number is None:
        print("추출된 숫자가 없습니다.")
    elif removed_number - existing_number != 1:
        print("한 페이지만 넘기세요.")
    else:
        with open(file_path, 'w') as file:
            file.write(str(removed_number))
        print(recognized_text)
    recognized_text = ""

    for field in result['images'][0]['fields']:
        text = field['inferText']
        recognized_text += text + " "

    return recognized_text.strip()

def process_image_with_ocr(api_url, secret_key, image_path):
    files = [('file', open(image_path, 'rb'))]

    request_json = {
        'images': [{'format': 'jpg', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    headers = {
        'X-OCR-SECRET': secret_key,
    }

    response = requests.post(api_url, headers=headers, data=payload, files=files)
    result = response.json()

    img = cv2.imread(image_path)
    roi_img = img.copy()

    recognized_text = ""
    removed_number = None

    for field in result['images'][0]['fields']:
        text = field['inferText']
       
        # Check if the first word is a number
        if removed_number is None and text.strip().split()[0].isdigit():
            removed_number = int(text.strip().split()[0])
            continue

        # Check if the last word is a number
        if text.strip().split()[-1].isdigit():
            removed_number = int(text.strip().split()[-1])
            break

        recognized_text += text + " "
       
        vertices_list = field['boundingPoly']['vertices']
        pts = [tuple(vertice.values()) for vertice in vertices_list]
        topLeft = [int(_) for _ in pts[0]]
        topRight = [int(_) for _ in pts[1]]
        bottomRight = [int(_) for _ in pts[2]]
        bottomLeft = [int(_) for _ in pts[3]]

        cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)

    if removed_number is not None:
        # Remove the found number from recognized_text
        recognized_text = recognized_text.replace(str(removed_number), '', 1)

    file_path = 'C:/Users/hyemi/Desktop/number.txt'  # 페이지 저장할 장소
    with open(file_path, 'r') as file:
        existing_number = int(file.read())

    if removed_number is None:
        print("추출된 숫자가 없습니다.")
    elif removed_number - existing_number != 1:
        print("한 페이지만 넘기세요.")
    else:
        with open(file_path, 'w') as file:
            file.write(str(removed_number))
        print(recognized_text)
