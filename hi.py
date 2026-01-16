import cv2
import os
import time

def capture_and_save_image():
    # 웹캠 초기화 (기본적으로 첫 번째 웹캠을 사용)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    # 사진 저장 경로 설정
    save_path = "./page_capture.jpg"

    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()

            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # 이전 사진 파일이 존재하면 삭제
            if os.path.exists(save_path):
                os.remove(save_path)

            # 현재 프레임을 저장
            cv2.imwrite(save_path, frame)
            print(f"{save_path}에 사진이 저장되었습니다.")

            # 5초 대기
            time.sleep(5)

    except KeyboardInterrupt:
        print("캡처가 중단되었습니다.")

    finally:
        # 웹캠 자원 해제
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_image()