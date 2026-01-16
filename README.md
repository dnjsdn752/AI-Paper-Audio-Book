# AI Paper Audio Book (시각장애인을 위한 AI 점자책 리더)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/H%2FW-Jetson%20Nano-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![YOLOv5](https://img.shields.io/badge/AI-YOLOv5-yellow)](https://github.com/ultralytics/yolov5)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red?logo=opencv&logoColor=white)](https://opencv.org/)
[![OCR](https://img.shields.io/badge/OCR-Naver%20CLOVA%20%7C%20EasyOCR-00C73C)](https://www.ncloud.com/)
[![TTS](https://img.shields.io/badge/TTS-Google%20Cloud-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/text-to-speech)
[![Captioning](https://img.shields.io/badge/Captioning-SK%20Cloud%20%7C%20MS%20GIT-orange)](#)

**AI Paper Audio Book**은 시각장애인이 종이책을 독립적으로 읽을 수 있도록 돕는 **임베디드 AI 솔루션**입니다. 카메라를 통해 책을 인식하고, **YOLOv5 객체 탐지, OCR, 이미지 캡셔닝** 기술을 유기적으로 결합하여 책의 내용을 사람의 목소리로 생생하게 읽어줍니다.

<div align="center" style="display: flex; justify-content: center; align-items: flex-start; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/aa603397-2fb9-401d-be49-faaa51678bae" width="48%" alt="시스템 이미지 1">
  <img src="https://github.com/user-attachments/assets/d7f1e4fc-440a-4018-b505-37fe290d816c" width="48%" alt="시스템 이미지 2">
</div>

---

## 📖 프로젝트 개요 (Overview)

대한민국 시각장애인의 90% 이상이 후천적 장애인 만큼 그들은 시각장애를 앓기 전 사물의 이미지에 대한 기억을 지니고 있습니다. 책에는 내용과 직접적인 연관이 있는 이미지가 많기에 이미지에 대한 전달이 필수적입니다. 그럼에도 시각장애인이 책을 읽을 수 있는 수단 중 가장 대표적인 점자책과 전자책은 완벽한 해결책이 되지 못합니다. 점자책의 경우 시각장애인의 5%만이 점자를 해독할 수 있고, 책 내부의 점자 오류가 많습니다. 전자책 어플의 경우 10만 권 정도의 책을 제공하나 책 내부의 이미지에 대한 설명이 없고 책이 아닌 서류나 문서를 읽고 싶은 경우 소용이 없습니다. 따라서 이미 집에 종이책이 구비되어 있는 시각장애인들에게 비용적 부담이 없이 책을 읽게 해주고, 책의 모든 내용을 이해할 수 있게 해주는 시스템이 필요합니다. 
이 프로젝트는 일반 도서를 **실시간으로 인식하여 오디오북처럼 읽어주는 시스템**을 구현함으로써 정보 접근성을 획기적으로 향상시킵니다.

### 핵심 목표
- **접근성(Accessibility)**: 복잡한 터치스크린이나 메뉴 조작 없이, **물리적 상호작용과 음성 피드백**만으로 시스템의 모든 기능을 통제할 수 있는 시각장애인 특화 UX를 제공합니다.
  
- **정확성(Accuracy)**: 책의 펼침에 따른 곡면 왜곡과 복잡한 외부 환경에서도 제본선과 텍스트 영역을 정교하게 분리하여, **누락이나 오독 없는 정확한 낭독**을 보장합니다.
  
- **연속성(Continuity)**: 책 표지와 마지막 페이지를 시스템이 기억하여, 언제든 **독서가 중단된 시점부터 자연스럽게** 이야기를 이어갈 수 있습니다. 비동기 오디오 설계를 통해 AI 연산 대기 시간에도 안내음을 제공하여 **청각적 끊김 없는 매끄러운 흐름**을 보장합니다.

---

## 🚀 주요 기능 (Key Features)

### 1. 스마트 책 인식 (Smart Book Recognition)
- **자동 식별**: 책 표지를 보여주면 이전에 읽던 책인지 자동으로 판별합니다.
- **이어 읽기**: 지난번에 멈춘 페이지를 기억했다가, "지난번에 50페이지까지 읽으셨습니다. 51페이지를 펼쳐주세요"라고 안내합니다.

### 2. 지능형 페이지 네비게이션 (Intelligent Navigation)
- **실시간 가이드**: 사용자가 페이지를 너무 많이 넘기거나(2장 이상), 덜 넘겼을 때 음성으로 피드백을 줍니다.
- **오차 보정**: 홀수/짝수 페이지 규칙을 이용하여 OCR 오인식률을 최소화합니다.

### 3. 멀티모달 독서 (Multimodal Reading)
- **텍스트 낭독 (OCR)**: 고성능 **AI OCR 모델**을 활용하여 책 본문의 텍스트를 정확하게 추출하고 인식합니다.
- **그림 설명 (Image Captioning)**: 책 내의 삽화나 도표가 발견되면, **Vision-Language 모델**이 이를 분석하여 "현재 페이지에는 ~한 그림이 있습니다"라고 상황을 묘사합니다.

<img width="1856" height="792" alt="image" src="https://github.com/user-attachments/assets/bc6c5c8e-0e99-47d9-8d4d-e0d9808eeb7c" />

---

## 🛠 시스템 구조 및 알고리즘 (System Logic)

이 시스템은 **Jetson Nano (Ubuntu)** 환경에서 단독으로 동작하도록 설계되었습니다.

### 1. 하드웨어 자동화 (Hardware Automation)
- **Startup**: 물리 버튼 클릭 시 Shell Script가 트리거되어 시스템 부팅 및 AI 애플리케이션(`main.py`) 자동 실행.
- **Shutdown**: 독서가 종료되면 소프트웨어적으로 `sudo shutdown now`를 호출하여 기기 전원 자동 차단.

### 2. 책 인식 알고리즘 (Book Recognition with ORB)
- **ORB Feature Matching**: 현재 웹캠에 비친 표지와 데이터베이스에 저장된 표지의 특징점을 비교합니다.
- **Resolution Optimization**: 4K(3840x2160) 고해상도 입력 영상을 유사도 계산 시에는 **1080p로 다운스케일링**하여, 임베디드 환경에서의 연산 속도와 매칭 정확도의 균형을 맞췄습니다.
- **Persistent Memory**: 책 제목과 마지막 페이지 정보를 **이미지 파일명**(`Title_PageNum.jpg`) 자체에 메타데이터로 인코딩하여 저장합니다. 이를 통해 별도 DB 없이도 전원 재부팅 시 상태를 완벽하게 복원합니다.

<img width="1900" height="614" alt="image" src="https://github.com/user-attachments/assets/8b562887-7a26-402f-adad-8f1925bdcc9a" />


### 3. 정밀 페이지 분할 (YOLOv5 & Center Line)
책이 기울어지거나 완전히 평평하지 않아도 정확하게 인식하기 위해 **구조적 객체 탐지**를 수행합니다.

- **YOLOv5 Classes**:
    - `m` (Center Line): 책의 제본선(중심).
    - `b` (Book): 책 전체 영역.
    - `Left` / `Right`: 왼쪽/오른쪽 페이지 텍스트 영역.
    - `LP` / `RP`: 페이지 번호 영역.
    - `IM`: 삽화(그림) 영역.

- **Dynamic Crop Algorithm**:
    - 탐지된 `m`(제본선) 좌표를 기준으로 `b`(책) 영역을 동적으로 분할합니다.
    - 이 방식은 단순 화면 반분할보다 **곡면이 있는 두꺼운 책** 인식에 훨씬 강력합니다.

<img width="1320" height="559" alt="image" src="https://github.com/user-attachments/assets/05fdacb4-ecbf-4792-bdfd-bbf0a9817a5e" />


### 4. 사용자 경험을 위한 비동기 오디오 (Asynchronous Audio UX)
시각장애인 사용자에게 "소리"는 시스템의 상태를 알 수 있는 유일한 피드백 수단입니다. 따라서 AI 연산 중에도 **끊김 없는 청각적 경험(Auditory Experience)**을 제공하는 것이 필수적입니다.

- **Non-blocking Audio Threading**:
    - YOLOv5 객체 탐지나 OCR 서버 통신과 같은 **Heavy Computing 작업**이 수행되는 동안, 메인 프로세스를 멈추지(Blocking) 않고 별도의 스레드(`threading.Thread`)에서 안내음(예: "인식 중입니다...", "잠시만 기다려주세요")을 재생합니다.
    - 이를 통해 사용자는 **시스템이 멈춘 것이 아니라 열심히 작업 중임**을 직관적으로 인지할 수 있으며, 체감 대기 시간을 획기적으로 줄였습니다.
- **Latency Masking**:
    - 이미지 캡셔닝이나 텍스트 변환에 3~5초 이상 소요될 때, *Process Start* 시점에 즉시 효과음을 재생하여 상호작용의 공백을 메웁니다.


---

## 💻 기술 스택 (Tech Stack)

| 구분 (Category) | 모델 / 기술 (Models & Tech) | 비고 (Note) |
| --- | --- | --- |
| **Language** | Python 3.9+ | |
| **H/W Platform** | **NVIDIA Jetson Nano** (Ubuntu) | Embedded System |
| **Object Detection** | **YOLOv5** (Custom Trained) | `best.pt` (Page), `best_cover.pt` (Cover) |
| **Computer Vision** | **OpenCV** (ORB Algorithm) | Feature Matching, Image Preprocessing |
| **OCR (Text)** | **Naver CLOVA OCR** (Online)<br>**EasyOCR** (Offline) | Main Architecture<br>Experimental Backup |
| **Image Captioning** | **SK Cloud API** (Online)<br>**Microsoft GIT** (Offline) | Main Architecture<br>On-device Research (`microsoft/git-base-coco`) |
| **TTS (Voice)** | **Google Cloud TTS** (WaveNet) | High-quality Neural Voice |

---

## 🔧 설치 및 실행 (Installation)

### 1. 환경 설정
```bash
# Repository 복제
git clone https://github.com/dnjsdn752/AI-Paper-Audio-Book.git
cd AI-Paper-Audio-Book

# 의존성 라이브러리 설치
pip install -r requirements.txt
```

### 2. API 키 설정
`secrets.json` 파일을 생성하여 API 키를 입력합니다. (보안을 위해 코드 분리됨)
```json
{
    "api_url": "YOUR_NAVER_OCR_URL",
    "secret_key": "YOUR_NAVER_SECRET_KEY",
    "caption_api_key": "YOUR_SK_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS": {
        "type": "service_account",
        ...
    }
}
```

### 3. 실행
개발 환경에서는 다음 명령어로 실행하며, 실제 하드웨어에서는 부팅 시 스크립트로 자동 실행됩니다.
```bash
python main.py
```

---

## 📂 폴더 구조 (Directory Structure)

```
📦 AI-Paper-Audio-Book
 ┣ 📂 yolov5/            # YOLOv5 객체 탐지 엔진
 ┣ 📂 mp3/               # 상황별 안내 음성 파일 (부팅, 에러, 가이드)
 ┣ 📂 metadata/          # (Deleted) 보안을 위해 secrets.json으로 통합됨
 ┣ 📂 TestUtils/         # 오프라인 모델(EasyOCR, MS GIT) 실험 코드
 ┣ 📜 main.py            # 메인 애플리케이션 (System Controller)
 ┣ 📜 Api_ocr.py         # OCR 통신 및 페이지 처리 로직
 ┣ 📜 config.py          # 환경변수 및 비밀키 로더
 ┣ 📜 secrets.json       # API 자격 증명 (Git Ignored)
 ┗ 📜 requirements.txt   # 의존성 패키지 목록
```
