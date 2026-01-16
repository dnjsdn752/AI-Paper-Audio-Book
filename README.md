# AI Paper Audio Book (시각장애인을 위한 AI 점자책 리더)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![YOLOv5](https://img.shields.io/badge/AI-YOLOv5-green)](https://github.com/ultralytics/yolov5)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Nano%20(Ubuntu)-orange)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![AI](https://img.shields.io/badge/Tech-Deep%20Learning%20OCR%20%26%20Captioning-red)](#)

**AI Paper Audio Book**은 시각장애인이 종이책을 독립적으로 읽을 수 있도록 돕는 **임베디드 AI 솔루션**입니다. 카메라를 통해 책을 인식하고, **딥러닝 기반의 객체 탐지, OCR, 이미지 캡셔닝 기술**을 복합적으로 활용하여 책의 내용을 사람의 목소리로 생생하게 읽어줍니다.

<!-- [추천 이미지 배치]: 시스템 전체 구성도나 실제 하드웨어(Jetson Nano + 카메라 + 스피커)가 설치된 사진을 여기에 넣으면 좋습니다. -->
<!-- ![System Overview](path/to/system_image.jpg) -->

---

## 📖 프로젝트 개요 (Overview)

기존의 점자책은 보급률이 낮고 제작 비용이 비쌉니다. 이 프로젝트는 일반 도서를 **실시간으로 인식하여 오디오북처럼 읽어주는 시스템**을 구현함으로써 정보 접근성을 획기적으로 향상시킵니다.

### 핵심 목표
- **접근성(Accessibility)**: 물리 버튼 하나로 켜고 꺼지는 직관적인 UX.
- **정확성(Accuracy)**: 제본선(Center Line) 인식을 통한 정교한 페이지 분할.
- **연속성(Continuity)**: 읽던 페이지를 기억하고 이어서 읽어주는 영속적 메모리.

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

<!-- [추천 이미지 배치]: 텍스트 읽기와 그림 설명이 동시에 작동하는 시연 스크린샷이나 다이어그램 추천 -->
<!-- ![Multimodal Demo](path/to/demo_image.jpg) -->

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

### 4. 사용자 경험을 위한 비동기 오디오 (Asynchronous Audio UX)
시각장애인 사용자에게 "소리"는 시스템의 상태를 알 수 있는 유일한 피드백 수단입니다. 따라서 AI 연산 중에도 **끊김 없는 청각적 경험(Auditory Experience)**을 제공하는 것이 필수적입니다.

- **Non-blocking Audio Threading**:
    - YOLOv5 객체 탐지나 OCR 서버 통신과 같은 **Heavy Computing 작업**이 수행되는 동안, 메인 프로세스를 멈추지(Blocking) 않고 별도의 스레드(`threading.Thread`)에서 안내음(예: "인식 중입니다...", "잠시만 기다려주세요")을 재생합니다.
    - 이를 통해 사용자는 **시스템이 멈춘 것이 아니라 열심히 작업 중임**을 직관적으로 인지할 수 있으며, 체감 대기 시간을 획기적으로 줄였습니다.
- **Latency Masking**:
    - 이미지 캡셔닝이나 텍스트 변환에 3~5초 이상 소요될 때, *Process Start* 시점에 즉시 효과음을 재생하여 상호작용의 공백을 메웁니다.

<!-- [추천 이미지 배치]: YOLOv5가 m(제본선), Left, Right 등을 바운딩 박스로 잡은 디텍션 결과 이미지 -->
<!-- ![YOLO Detection](path/to/yolo_result.jpg) -->

---

## 💻 기술 스택 (Tech Stack)

| 구분 | 기술 / 모델 (Technology & Models) |
| --- | --- |
| **Language** | Python 3.9+ |
| **Object Detection** | YOLOv5 (Custom Trained: `best.pt`, `best_cover.pt`) |
| **Computer Vision** | OpenCV (ORB Feature Matching), NumPy |
| **OCR (Text)** | Deep Learning OCR & Scene Text Recognition |
| **TTS (Voice)** | Neural Text-to-Speech (WaveNet) |
| **Captioning** | Image Captioning Transformer |
| **Platform** | NVIDIA Jetson Nano (Ubuntu 18.04/20.04) |

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
