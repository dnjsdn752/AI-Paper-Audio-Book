
import os
import glob
from Api_ocr import *
from config import API_URL, SECRET_KEY

'''
search_im = './runs/detect/exp/crops/IM'
search_im_path = '.runs/detect/exp/crops/IM/*'
search_LP = '.runs/detect/exp/crops/LP'
search_RP = '.runs/detect/exp/crops/RP'



# 현재 디렉토리의 .jpg 파일 목록 확인
print(glob.glob(search_im_path))
'''
'''
import os
from PIL import Image

# 폴더 경로
folder_path ='./runs/detect/exp/crops/IM'
jpg_file = 'page_capture0.jpg'
jpg_img_path = os.path.join('./runs/detect/exp/crops/IM', jpg_file)
print(jpg_img_path)
'''

search_im = './runs/detect/exp/crops/IM'
search_im_path = './runs/detect/exp/crops/IM/*.jpg'
search_LP = './runs/detect/exp/crops/LP'
search_RP = './runs/detect/exp/crops/RP'
api_url = API_URL
secret_key = SECRET_KEY

if os.path.isdir(search_RP):
    print('ss')
    search_RP_path = './runs/detect/exp/crops/RP/page_capture0.jpg'
    print('ss')
    page_number_detection = False
          
    current_number_RP = Api_ocr_image(api_url, secret_key, search_RP_path)  
    print("detect_RP")
                    
    print(current_number_RP)
                    
    if current_number_RP.isdigit():
        print('hi')
        current_number = int(current_number_RP)
#왼쪽페이지 인식
if os.path.isdir(search_LP):
    search_LP_path = './runs/detect/exp/crops/LP/page_capture0.jpg'
    current_number_LP = Api_ocr_image(api_url, secret_key, search_LP_path)
    print("detect_LP")
    print(current_number_LP)
    if current_number_LP.isdigit():
        page_number_detection = True
        current_number = int(current_number_LP) + 1   
Left_text, current_number_LP = Api_ocr_process_image_with_page_number(api_url, secret_key, './runs/detect/exp/crops/Left/page_capture.jpg')
Right_text, current_number_RP = Api_ocr_process_image_with_page_number(api_url, secret_key, './runs/detect/exp/crops/Right/page_capture.jpg')
print(Left_text)
print(current_number_LP)
print(Right_text)
print(current_number_RP)
Left_text = Api_ocr_image(api_url, secret_key, './runs/detect/exp/crops/Left/page_capture.jpg')
Right_text = Api_ocr_image(api_url, secret_key, './runs/detect/exp/crops/Right/page_capture.jpg')
print(Left_text)
print(Right_text)