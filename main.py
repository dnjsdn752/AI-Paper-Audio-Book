import cv2
import numpy as np
#from gtts import gTTS
from playsound import playsound
from PIL import Image, ImageDraw, ImageFont 
#from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import matplotlib.pyplot as plt
#from googletrans import Translator
import subprocess
import time
import os
#import speedtest
from Api_ocr import *
from config import API_URL, SECRET_KEY, CAPTION_API_KEY, CAPTION_URL_BASE, GOOGLE_APPLICATION_CREDENTIALS
#import deepl
import glob
import threading
from google.cloud import texttospeech
import torch
import pathlib
import shutil
from collections import defaultdict
from pathlib import Path

def play_sound(file_path):
    playsound(file_path)

# MP3 파일 재생 (비동기적으로 실행)
sound_thread = threading.Thread(target=play_sound, args=('./mp3/booting.mp3',))
sound_thread.start()

# Example usage: ocr
# Example usage: ocr
api_url = API_URL
secret_key = SECRET_KEY

#translation
#auth_key = "77a59737-132b-4dcf-86a8-2a8a6888c1c0:fx"


# 발급받은 API 키 captioning
caption_api_key = CAPTION_API_KEY
caption_url = f'{CAPTION_URL_BASE}?appkey={caption_api_key}'

# Creates a client tts
if GOOGLE_APPLICATION_CREDENTIALS and isinstance(GOOGLE_APPLICATION_CREDENTIALS, dict):
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_info(GOOGLE_APPLICATION_CREDENTIALS)
    client = texttospeech.TextToSpeechClient(credentials=credentials)
else:
    client = texttospeech.TextToSpeechClient()

# Select the language and SSML Voice Gender (optional)
voice = texttospeech.VoiceSelectionParams(
    language_code='ko-KR',
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    name='ko-KR-Wavenet-B'
)

# Select the type of audio encoding
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

#yolo model load
pathlib.WindowsPath = pathlib.PosixPath
save_dir = './runs/detect/exp/crops'
yolo_model = torch.hub.load('./yolov5', 'custom', path= './best.pt', source='local')
yolo_model.conf = 0.5
yolo_cover_model = torch.hub.load('./yolov5', 'custom', path= './best_cover.pt', source='local')
yolo_cover_model.conf = 0.5

'''
#captioning
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
'''

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

def yolo_detect_save(image_path):
    m_center = None
    out = yolo_model(image_path)

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

def yolo_cover_detect_save(image_path):
    m_center = None
    out = yolo_cover_model(image_path)

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

def gc_speak(txt):
    if txt is not None:

        # The text to synthesize
        text = txt

        # Construct the request
        input_text = texttospeech.SynthesisInput(text=text)

        # Performs the Text-to-Speech request
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )

        if isinstance(txt, str):  # 문자열인 경우에만 음성 출력
            # Save the audio response to a temporary file
            temp_file = 'temp_audio.mp3'
            if os.path.exists(temp_file):
                os.remove(temp_file)
            with open(temp_file, 'wb') as out:
                out.write(response.audio_content)
                
            playsound(temp_file)

            '''
            # Play the audio using default player
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['afplay', temp_file])
            elif platform.system() == 'Windows':  # Windows
                os.startfile(temp_file)
            elif platform.system() == 'Linux':  # Linux
                subprocess.run(['aplay', temp_file])
            else:
                print("Unsupported operating system. Cannot play audio automatically.")
            '''

def speak(text_t, file_name):
    if os.path.exists(file_name): #시작메시지파일이 있는 경우 삭제하고 다시 생성 (권한 문제)
        os.remove(file_name)
    tts = gTTS(text=text_t, lang='ko')
    tts.save(file_name)
    playsound(file_name)

def page_number_check(numbers, previous_number):
    page_change = True
    page_check = False
    
    if int(numbers) > int(previous_number) + 2:
        difference = int(numbers) - int(previous_number)
        error_message = f" 현재{numbers}페이지입니다. 앞으로 {round(difference/2)-1}장만 넘기세요."
        gc_speak(error_message)
        
    elif int(numbers) == int(previous_number) + 2:
        page_check = True
        
    elif int (numbers) == int(previous_number):
        error_message = f" 현재 페이지 변화가 감지 되지 않고 있습니다. 페이지를 넘겨주세요. 세번 경고 이후엔 자동으로 프로그램이 종료됩니다."
        gc_speak(error_message)
        page_change = False
        
    elif int(numbers) < int (previous_number):
        difference = int(previous_number) - int(numbers)
        error_message = f" 현재{numbers}페이지입니다. 뒤로 {round(difference/2)+1}장만 넘기세요."
        gc_speak(error_message)
   
    return page_check, page_change

def no_api_captioning(image):

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    caption_text = generated_caption
    print(caption_text)
    translator = Translator()
    captioning_text = (translator.translate(caption_text, dest='ko').text)
    
    return captioning_text

def api_captioning(image):

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    caption_text = generated_caption
    print(caption_text)
    deep_translator = deepl.Translator(auth_key)
    captioning_text = (deep_translator.translate_text(caption_text, target_lang="ko").text)
    
    return captioning_text

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

def calculate_image_similarity_percentage(image_1, image_2):
   
    # ORB 감지기 초기화
    orb = cv2.ORB_create()

    # 이미지에서 키포인트와 디스크립터 찾기
    kp1, des1 = orb.detectAndCompute(image_1, None)
    kp2, des2 = orb.detectAndCompute(image_2, None)

    # BFMatcher 객체 생성
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 디스크립터를 매칭하고 거리 순으로 정렬
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 유사도 계산 (매칭된 키포인트 수 / 전체 키포인트 수)
    similarity_percentage = (len(matches) / max(len(kp1), len(kp2))) * 100

    return similarity_percentage

def compare_with_title_images(cam_image, title):
    
    folder_path = "./title"
    files = os.listdir(folder_path)
    check = 0
    
    if files:        
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            
            try:
                folder_image = cv2.imread(file_path)
                
                similarity = calculate_image_similarity_percentage(cam_image, folder_image)
                print(similarity)
                if similarity >= 40:
                    # 먼저 파일명에서 확장자를 제거합니다.
                    file_name_without_extension = file_name.rsplit('.', 1)[0]

                    # 띄어쓰기를 기준으로 나눕니다.
                    words = file_name_without_extension.split(' ')

                    # 마지막 단어를 추출합니다.
                    page = int(words[-1])
                    gc_speak(title + "은 이전에 읽은 기록이 있습니다. " + str(page+1) + "페이지를 펼쳐주세요.  10초 후에 책을 읽어 드리겠습니다.")
                    check = 1
                    title_name = file_name
                    break
                
            except Exception as e:
                print(f"{file_name} 이미지 처리 중 오류 발생: {e}")
                
    if check==0 :      
        gc_speak(title + " 책 제목은 기록에서 발견하지 못하였습니다. 새로운 책으로 등록하겠습니다.  10초 후에 책을 읽어 드리겠습니다.")        
        image_name = time.strftime("%Y%m%d-%H%M%S") + " 0" + ".jpg"
        image_path = os.path.join(folder_path, image_name)
    
        # 이미지 저장
        cv2.imwrite(image_path, cam_image)  # RGB에서 BGR로 변환하여 저장
        page = 0
        title_name = image_name
    
    return page, title_name

def check_internet_speed():
    st = speedtest.Speedtest()
    st.download()  # 또는 st.upload()를 사용하여 업로드 속도 측정
    st.upload()
    download_speed = st.results.download / 1_000_000  # Mbps 단위로 변경
    upload_speed = st.results.upload / 1_000_000  # Mbps 단위로 변경
    print(f"다운로드 속도: {download_speed} Mbps, 업로드 속도: {upload_speed} Mbps")
    return download_speed, upload_speed

def check_internet_connection():
    connection = 0
    try:
        # Google 메인 페이지에 대한 GET 요청을 시도합니다.
        response = requests.get('http://www.google.com', timeout=5)
        # 요청이 성공적으로 완료되면, 인터넷 연결이 있다고 판단합니다.
        connection = 1 #연결되면 1
        speak("인터넷에 연결되어 있습니다.", "hello.mp3")
    except requests.ConnectionError as e:
        # 연결 오류가 발생하면, 인터넷 연결이 없다고 판단합니다.
        speak("인터넷에 연결되어 있지 않습니다.", "hello.mp3")
        connection = 0
    return connection    

def change_last_character(subdir, file_name, new_char):
    # 파일의 전체 경로를 생성합니다.
    file_path = os.path.join(subdir, file_name)
    
    # 파일 이름의 확장자를 분리합니다.
    name_part, extension = os.path.splitext(file_name)
    
    # 파일 이름을 띄어쓰기를 기준으로 단어로 나눕니다.
    words = name_part.split(' ')
    
    # 마지막 단어의 마지막 글자를 새로운 글자로 변경합니다.
    print(words[-1])
    words[-1] = str(new_char)
    print(words[-1])
    
    # 변경된 단어들을 다시 띄어쓰기로 연결하고 확장자를 추가합니다.
    new_name = ' '.join(words) + extension
    
    # 새로운 파일의 전체 경로를 생성합니다.
    new_path = os.path.join(subdir, new_name)
    
    # 파일 이름을 변경합니다.
    os.rename(file_path, new_path)

    return new_name
    



#실행 코드
'''
speak('네트워크 환경을 확인 중입니다.', 'hello.mp3')
#인터넷연결 되어있는지 먼저 확인
network = check_internet_connection()
MIN_DOWNLOAD_SPEED = 10
MIN_UPLOAD_SPEED = 5
if network ==1:
    #인터넷속도에 따라서 동작하는 코드 다르게 설정 (수정필요)
    download_speed, upload_speed = check_internet_speed()

    if download_speed >= MIN_DOWNLOAD_SPEED and upload_speed >= MIN_UPLOAD_SPEED:
        speak("인터넷 속도가 안정적입니다.", "hello.mp3")
        version = 'api_version'
    else:
        speak("인터넷 속도가 불안정하여 demo 버전으로 전환합니다.", "hello.mp3")
        version = 'no_api_version'
else:
    speak('인터넷이 연결 되지 않아 demo 버전으로 전환합니다.', "hello.mp3")
    version = 'no_api_version'
'''    

search_im = './runs/detect/exp/crops/IM'
search_im_path = './runs/detect/exp/crops/IM/*.jpg'
search_LP = './runs/detect/exp/crops/LP'
search_RP = './runs/detect/exp/crops/RP'


version = 'api_version'
#Api version 
sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림
if version=='api_version':
    
    # MP3 파일 재생 (비동기적으로 실행)
    sound_thread = threading.Thread(target=play_sound, args=('./mp3/camera.mp3',))
    sound_thread.start()
    #웹캠 연결
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림
    
    #소리 출력 (책 제목을 보여주세요)
    play_sound("./mp3/start_guide.mp3")

    title_attempt = 0

    #5번 시도 안에 성공 못할시 종료
    while title_attempt < 5:

        # MP3 파일 재생 (비동기적으로 실행)
        sound_thread = threading.Thread(target=play_sound, args=('./mp3/cover_ocr.mp3',))
        sound_thread.start()    
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        
        if not ret:
            break
        
        image_path = "./page_capture.jpg"
        if os.path.exists(image_path):
            os.remove(image_path)
        if ret:
            cv2.imwrite(image_path, frame)
       
        #yolo cover detect
        yolo_cover_detect_save(image_path)
        detect_cover_path = './runs/detect/exp/crops/0/page_capture0.jpg'
        
        if os.path.exists(detect_cover_path):
            title = Api_ocr_image(api_url, secret_key, detect_cover_path)
            cover_detection = True
        else:
            title = Api_ocr_image(api_url, secret_key, image_path)
            cover_detection = False
        sound_thread.join()

        # 텍스트가 인식되었다면 출력하고 재시도 횟수 초기화
        if title.strip():
            gc_speak("성공")
            
            #더 나은 유사도 판정을 위해 1080p로 다운그레이드
            image_path = "./page_capture.jpg"
            
            resized_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            if os.path.exists(image_path):
                os.remove(image_path)
            if ret:
                cv2.imwrite(image_path, resized_frame)
            cap_img = cv2.imread(image_path)

            # MP3 파일 재생 (비동기적으로 실행)
            #sound_thread = threading.Thread(target=play_sound, args=('./mp3/cover_captioning.mp3',))
            #sound_thread.start()
            if cover_detection==True:
                title_caption = sk_api_captioning(detect_cover_path)
            else:
                title_caption = sk_api_captioning(image_path)

            #sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림

            gc_speak("책 표지 그림에 대하여 설명하겠습니다. " + title_caption) 
            
            
            #책 제목이 memory에 있는지 검색하고 있는 경우 speak 없을경우 memory에 책 제목 추가
            page, title_name = compare_with_title_images(cap_img, title)
            
            #페이지를 찾을 때 까지 10초의 시간
            time.sleep(10)
            
            #파일이름으로부터 마지막 페이지를 가져온다.
            title_name_without_extension = title_name.rsplit('.', 1)[0]
            words = title_name_without_extension.split(' ')
            previous_number = int(words[-1])
            
            #처음 읽는 책일 경우 처음에 어떻게 처리할지 if 문
            if previous_number == 0:
                book_error_count = 0
                play_sound("./mp3/first_page_number_guide.mp3")
                while True:
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    time.sleep(1)
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    ret, frame = cap.read()
                    ret, frame = cap.read()

                    #기존 사진 삭제 후 현재 화면 저장 
                    image_path = "./page_capture.jpg"
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    if ret:
                        cv2.imwrite(image_path, frame)
                    
                    # MP3 파일 재생 (비동기적으로 실행)
                    sound_thread = threading.Thread(target=play_sound, args=('./mp3/detection_loading.mp3',))
                    sound_thread.start()
                    #yolo로 detection한 이미지들
                    yolo_detect_save('./page_capture.jpg')
                    print("first")
                    sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림

                    if not os.path.isdir('./runs/detect/exp/crops/b'):
                        book_error_count = book_error_count + 1
                        if book_error_count < 3:
                            play_sound("./mp3/book_error.mp3")
                        else:
                            play_sound("./mp3/book_error_skip.mp3")    
                            book_error_count = 0
                            time.sleep(10)
                        continue

                    # MP3 파일 재생 (비동기적으로 실행)
                    #sound_thread = threading.Thread(target=play_sound, args=('./mp3/ocr_loading.mp3',))
                    #sound_thread.start()
                    #ocr
                    Left_text, current_number_LP = Api_ocr_process_image_with_page_number(api_url, secret_key, './runs/detect/exp/crops/Left/page_capture.jpg')
                    Right_text, current_number_RP = Api_ocr_process_image_with_page_number(api_url, secret_key, './runs/detect/exp/crops/Right/page_capture.jpg')
                    #sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림

                    if os.path.isdir(search_im):
                        print("dddd")    
                        jpg_files = [f for f in os.listdir(search_im) if f.endswith('.jpg')]
                        for jpg_file in jpg_files:
                            print("xxxxx")
                            print(jpg_file)
                            # MP3 파일 재생 (비동기적으로 실행)
                            #sound_thread = threading.Thread(target=play_sound, args=('./mp3/content_captioning.mp3',))
                            #sound_thread.start()
                            jpg_img_path = os.path.join('./runs/detect/exp/crops/IM', jpg_file)
                            
                            image_caption = sk_api_captioning(jpg_img_path)
                            #sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림
                            gc_speak("현재 페이지에 있는 그림에 대하여 설명해드리겠습니다. " +image_caption )
                        gc_speak("   이어서 책을 읽어 드리겠습니다.   " + Left_text )
                    else:
                        gc_speak("   " + Left_text)
                        
                    #오른쪽페이지
                    gc_speak("   " + Right_text + "   책을 한장 넘겨주세요.")
                    
                    book_error_count = 0
                    time.sleep(10)    
                        
                    #책 오른쪽 페이지가 인식 되었는지 확인
                    if os.path.isdir(search_RP):
                        search_RP_path = './runs/detect/exp/crops/RP/page_capture0.jpg'
                        current_number = Api_ocr_image(api_url, secret_key, search_RP_path)
                        print(" page right detection" + current_number)
                        if current_number.isdigit():
                           
                           if int(current_number) % 2 != 0:
                                print(current_number)
                                title_name = change_last_character('./title', title_name, current_number)
                                previous_number = current_number
                                play_sound('./mp3/first_page_number_detection.mp3')
                                break  
                    #왼쪽페이지 인식
                    if os.path.isdir(search_LP):
                        search_LP_path = './runs/detect/exp/crops/LP/page_capture0.jpg'
                        current_number = Api_ocr_image(api_url, secret_key, search_LP_path)
                        print(" page left detection" + current_number)
                        if current_number.isdigit():
                            if int(current_number) % 2 == 0:
                                current_number = int(current_number) + 1
                                print(current_number)
                                title_name = change_last_character('./title', title_name, current_number)
                                previous_number = current_number
                                play_sound('./mp3/first_page_number_detection.mp3')
                                break           
                    #LP, RP detection 실패 시 동작           
                    if current_number_RP is not None:
                        current_number = int(current_number_RP)  
                        title_name = change_last_character('./title', title_name, current_number)
                        previous_number = current_number
                        play_sound('./mp3/first_page_number_detection.mp3')
                        break
                    if current_number_LP is not None:
                        current_number = int(current_number_LP) + 1
                        title_name = change_last_character('./title', title_name, current_number)
                        previous_number = current_number
                        play_sound('./mp3/first_page_number_detection.mp3')
                        break
                        
            #페이지 변화 감지 3번 반복
            page_change_attempt = 0
            book_error_count = 0
            booting_page_check=1
            while page_change_attempt < 3:
                
                ret, frame = cap.read()
                ret, frame = cap.read()
                ret, frame = cap.read()
                ret, frame = cap.read()
                ret, frame = cap.read()
                time.sleep(1)
                ret, frame = cap.read()
                ret, frame = cap.read()
                ret, frame = cap.read()
                ret, frame = cap.read()
                ret, frame = cap.read()
                ret, frame = cap.read()
                        
                #기존 사진 삭제 후 현재 화면 저장 
                image_path = "./page_capture.jpg"
                if ret:
                    cv2.imwrite(image_path, frame)
                
                # MP3 파일 재생 (비동기적으로 실행)
                sound_thread = threading.Thread(target=play_sound, args=('./mp3/detection_loading.mp3',))
                sound_thread.start()

                #yolo로 detection한 이미지들
                yolo_detect_save('./page_capture.jpg')
                print("exp")
                
                sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림

                if not os.path.isdir('./runs/detect/exp/crops/b'):
                    book_error_count = book_error_count + 1
                    if book_error_count < 3:
                        play_sound("./mp3/book_error.mp3")
                    else:
                        play_sound("./mp3/book_error_skip.mp3")   
                        time.sleep(10)
                        previous_number = int(previous_number) + 2
                        book_error_count = 0
                    continue

                #페이지 수 인식
                page_number_detection = False
                if os.path.isdir(search_RP):
                    search_RP_path = './runs/detect/exp/crops/RP/page_capture0.jpg'
                    current_number_RP = Api_ocr_image(api_url, secret_key, search_RP_path)  
                    print("detect_RP")
                    
                    print(current_number_RP)
                    if current_number_RP.isdigit():
                        page_number_detection = True
                        current_number = int(current_number_RP)
                   
                #왼쪽페이지 인식
                elif os.path.isdir(search_LP):
                    search_LP_path = './runs/detect/exp/crops/LP/page_capture0.jpg'
                    current_number_LP = Api_ocr_image(api_url, secret_key, search_LP_path)
                    print("detect_LP")
                    if current_number_LP.isdigit():
                        page_number_detection = True
                        current_number = int(current_number_LP) + 1
                        
                #왼쪽 오른쪽 둘다 페이지 인식 실패시
                if (not os.path.isdir(search_RP) and not os.path.isdir(search_LP)) or page_number_detection == False:
                    page_number_detection = False
                    current_number = None
                    Left_text, current_number_LP = Api_ocr_process_image_with_page_number(api_url, secret_key, './runs/detect/exp/crops/Left/page_capture.jpg')
                    Right_text, current_number_RP = Api_ocr_process_image_with_page_number(api_url, secret_key, './runs/detect/exp/crops/Right/page_capture.jpg')
                    
                    if current_number_LP is not None:
                        current_number = int(current_number_LP) + 1
                    if current_number_RP is not None:
                        current_number = int(current_number_RP)  
                    if current_number is not None:    
                        print("no detect number")  
                        page_check, page_change = page_number_check(current_number, previous_number) 
                        #page 변화가 없다면 반복문 넘어감
                        if page_change==False:
                            page_change_attempt +=1
                            time.sleep(10)
                            continue
                        if page_check==False:
                            time.sleep(10)
                            continue
                    if current_number is None:
                        print("no_number")
                        if booting_page_check == 1:
                            play_sound("./mp3/booting_page_check_error.mp3")
                            continue
                        else:    
                            current_number = int(previous_number) + 2    
                    
                if page_number_detection == True:
                    if int(current_number)%2 ==0:
                        current_number=int(current_number)+1
                    print(previous_number)
                    print(current_number)
                    #previous는 오른쪽 페이지(홀수), current는 왼쪽페이지(짝수) 즉 무조건 current = previous+1 
                    page_check, page_change = page_number_check(current_number, previous_number) 
                    #page 변화가 없다면 반복문 넘어감
                    if page_change==False:
                        page_change_attempt +=1
                        time.sleep(10)
                        continue
                    if page_check==False:
                        time.sleep(10)
                        continue
                        
                    # MP3 파일 재생 (비동기적으로 실행)
                    #sound_thread = threading.Thread(target=play_sound, args=('./mp3/ocr_loading.mp3',))
                    #sound_thread.start()

                    #ocr
                    Left_text = Api_ocr_image(api_url, secret_key, './runs/detect/exp/crops/Left/page_capture.jpg')
                    Right_text = Api_ocr_image(api_url, secret_key, './runs/detect/exp/crops/Right/page_capture.jpg')   

                    #sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림

                if os.path.isdir(search_im):
                    print("dddd")    
                    jpg_files = [f for f in os.listdir(search_im) if f.endswith('.jpg')]
                    for jpg_file in jpg_files:
                        print("xxxxx")
                        print(jpg_file)
                        # MP3 파일 재생 (비동기적으로 실행)
                        #sound_thread = threading.Thread(target=play_sound, args=('./mp3/content_captioning.mp3',))
                        #sound_thread.start()
                        jpg_img_path = os.path.join('./runs/detect/exp/crops/IM', jpg_file)
                            
                        image_caption = sk_api_captioning(jpg_img_path)
                        #sound_thread.join()  # 사운드 재생이 끝날 때까지 기다림
                        gc_speak("현재 페이지에 있는 그림에 대하여 설명해드리겠습니다. " +image_caption )
                    gc_speak("   이어서 책을 읽어 드리겠습니다.   " + Left_text )
                else:
                    gc_speak("   " +Left_text )
                    
                #오른쪽페이지
                gc_speak("   " +Right_text )
                                
                #페이지 넘버 저장 (왼쪽 오른쪽 고려)
                title_name = change_last_character('./title', title_name, current_number)
                previous_number = current_number
                
                #다음페이지 안내 (왼쪽 오른쪽 고려)
                gc_speak("    다음 페이지로 넘겨주세요")
                page_change_attempt = 0
                book_error_count = 0
                booting_page_check = 0
                time.sleep(10)
            
            
            #읽어주는 반복문 종료 memory에 저장(페이지는 저장되어 있음)  
            gc_speak("프로그램을 종료합니다. ") 
            break
            
        # 텍스트가 인식되지 않았다면 에러 메시지 출력하고 재시도 횟수 증가
        else:
            gc_speak("책 제목을 찾지 못했습니다. 책을 정확하게 위치시켜주세요. " )
            title_attempt += 1
            
        # 일정 시간 간격마다 OCR 수행
        time.sleep(7)
    if title_attempt==5:
        gc_speak("책 제목을 찾지 못하여 프로그램을 종료합니다. ")

    cap.release()
    
#demo 버전
elif version=='no_api_version':
    #demo버전        
    speak('안해', 'hello.mp3')

os.system('sudo shutdown now')