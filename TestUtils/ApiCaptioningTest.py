import requests
import time
import os
from config import CAPTION_API_KEY, CAPTION_URL_BASE


# 발급받은 API 키 captioning
caption_api_key = CAPTION_API_KEY
caption_url = f'{CAPTION_URL_BASE}?appkey={caption_api_key}'

def sk_api_captioning(image_file_path):
    # 업로드할 이미지 파일 경로

    # 파일 읽기
    with open(image_file_path, 'rb') as image_file:
        files = {'image': image_file}
        # 요청 전송
        response = requests.post(caption_url, files=files)

    # 응답 처리
    if response.status_code == 200:
        # 캡션 결과 출력
        caption = response.json()['caption']
        return caption
    else:
        # 오류 메시지 출력
        print("오류 발생:", response.status_code, response.text)
        return None

start = time.time()
text = sk_api_captioning("./captured_image.jpg")
point1 = time.time()
print(f"{point1-start:.5f}sec")
print(text)