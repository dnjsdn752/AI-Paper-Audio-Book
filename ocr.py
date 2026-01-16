import cv2
import numpy as np
from gtts import gTTS
import playsound

# 이미지를 불러옵니다.
img = cv2.imread('C:/Users/min/Desktop/x.jpg')

# 검정색 범위를 정의합니다. 여기서는 RGB 값이 45~55인 픽셀을 검정색으로 정의했습니다.
lower_black = np.array([0, 0, 0])
upper_black = np.array([140, 140, 140])

# 이미지에서 검정색 부분을 마스킹합니다.
mask = cv2.inRange(img, lower_black, upper_black)

# 마스크를 적용하여 검정색 부분만 남기고 나머지 부분을 흰색으로 만듭니다.
result = cv2.bitwise_and(img, img, mask=mask)
result[mask==0] = [255, 255, 255]

# 결과 이미지를 저장합니다.
cv2.imwrite('C:/Users/min/Desktop/result.jpg', result)


import easyocr
import cv2
from PIL import Image, ImageDraw, ImageFont 

# EasyOCR Reader 생성
reader = easyocr.Reader(['ko'])

# 이미지에서 텍스트 추출
result = reader.readtext('C:/Users/min/Desktop/result.jpg')

# 이미지를 OpenCV로 불러오기
img = cv2.imread('C:/Users/min/Desktop/result.jpg')
img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

text_list = [res[1] for res in result]

# 폰트 설정 (예: Gulim 폰트, 글자 크기)
font = ImageFont.truetype("C:/Windows/Fonts/gulim.ttc", 24)

img = cv2.imread('C:/Users/min/Desktop/result.jpg')
img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)
for detection in result:
    box, _, _ = detection
    points = [tuple(map(int, point)) for point in box]
    draw.polygon(points, outline="red")

# Group boxes by y-coordinate similarity
grouped_boxes = {}
for box, text, _ in result:
    y_coordinates = [point[1] for point in box]
    min_box_y = min(y_coordinates[:2])

    # Search for similar y-coordinates within a range of 10 pixels
    found_group = False
    for key in grouped_boxes:
        if abs(min_box_y - key) <= 30:
            grouped_boxes[key].append((box, text))
            found_group = True
            break

    if not found_group:
        grouped_boxes[min_box_y] = [(box, text)]

# Find the box with the maximum y-coordinate within the groups
max_y = float('-inf')
max_y_boxes = []
max_texts = []
for key, boxes in grouped_boxes.items():
    for box, text in boxes:
        if key > max_y:
            max_y = key
            max_y_boxes = [box]
            max_texts = [text]
        elif key == max_y:
            max_y_boxes.append(box)
            max_texts.append(text)


# Merge boxes with the same maximum y-coordinate
if len(max_y_boxes) > 1:
    merged_box = [
        (min(box[0][0] for box in max_y_boxes), min(box[0][1] for box in max_y_boxes)),
        (max(box[2][0] for box in max_y_boxes), min(box[1][1] for box in max_y_boxes)),
        (max(box[1][0] for box in max_y_boxes), max(box[2][1] for box in max_y_boxes)),
        (min(box[3][0] for box in max_y_boxes), max(box[3][1] for box in max_y_boxes))
    ]
    
    max_y_boxes = [merged_box]

# Remove the text corresponding to the merged box from the text list
filtered_text_list = [line for line in text_list if line not in max_texts]

# Draw the merged box
draw = ImageDraw.Draw(img_pil)
numbers = ''.join(filter(str.isdigit, max_texts))

with open('C:/Users/min/Desktop/extracted_numbers.txt', 'r') as file:
    previous_number = file.read()

# Check if the new number is not equal to the previous number + 1
if int(numbers) != int(previous_number) + 1:
    print("Error: The extracted number is not incremented by 1.")
else:
    with open('C:/Users/min/Desktop/extracted_numbers.txt', 'w') as file:
        file.write(numbers)

    for box in max_y_boxes:
        points = [tuple(map(int, point)) for point in box[0:4]]
        draw.polygon(points, outline="blue")  # Change the outline color to blue for the merged box
        
    img_pil.save('C:/Users/min/Desktop/result_with_merged_box.jpg')

    # Print the filtered text list
    with open('C:/Users/min/Desktop/extracted_text.txt', 'w', encoding='utf-8') as file:
        for line in filtered_text_list:
            file.write(line + '\n')
    
    text = ''
    with open('C:/Users/min/Desktop/extracted_text.txt', 'r', encoding='utf-8') as file:
        text = file.read()

# 텍스트를 음성 파일로 변환
    file_name = "231018.mp3"
    tts = gTTS(text=text, lang="ko")
    tts.save(file_name)
    playsound.playsound(file_name)


    




