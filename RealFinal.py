import cv2
import torch
from pathlib import Path
import random
import serial
from tensorflow.keras.models import load_model
import pyaudio
import numpy as np
import wave
import librosa
import threading

# 아두이노와 연결된 시리얼 포트 설정 (적절한 포트 이름으로 변경하세요)
arduino_port = 'COM8'

# 시리얼 통신을 열고 아두이노와 연결합니다.
ser = serial.Serial(arduino_port, 9600)

# 모델을 로드할 때 이미 로드 되어 있는지 확인 하고, 없는 경우에만 로드 합니다.
if 'model' not in locals():
    model = load_model('C:/Users/hsw03/PycharmProjects/capstonedata_cnn_model/mfcc_model')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

# 모델 파일의 경로 설정 (적절한 경로로 변경하세요)
model_path = 'runs/train/exp1/weights/best.pt'

# YOLOv5 모델을 초기화합니다.
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 클래스별 랜덤 색상을 생성합니다.
num_classes = len(yolo_model.names)
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]

# 웹캠 비디오 스트림을 캡처합니다.
cap = cv2.VideoCapture(0)

# 객체 감지를 수행하는 함수
def object_detection():
    last_send_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 프레임을 YOLOv5 모델에 전달하여 객체 감지를 수행합니다.
        results = yolo_model(frame)

        # 현재 시간을 가져옵니다.
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # 13초마다 한 번씩 데이터를 아두이노로 전송합니다.
        if current_time - last_send_time >= 10:
            for detection in results.pred[0]:
                class_id = int(detection[-1].item())  # 클래스 ID를 추출합니다.
                class_name = yolo_model.names[class_id]  # 클래스명을 가져옵니다.
                confidence = detection[4].item()  # 신뢰도(확률)를 추출합니다.

                # 객체 감지 박스의 좌표를 추출합니다.
                bbox = detection[:4].tolist()
                x1, y1, x2, y2 = map(int, bbox)

                # 클래스별 랜덤 색상을 사용하여 객체 감지 박스를 그립니다.
                color = class_colors[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 클래스명과 신뢰도를 화면에 출력합니다.
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 클래스명과 신뢰도를 노트북 화면에 출력하거나 다른 작업을 수행합니다.
                print(f"Class: {class_name}, Confidence: {confidence}")

                # 클래스명과 신뢰도를 아두이노로 전송합니다.
                data = f"Class: {class_name}, Confidence: {confidence}\n"
                ser.write(data.encode())  # 데이터를 아두이노로 전송합니다.
                last_send_time = current_time  # 마지막 전송 시간 업데이트

        # 결과를 화면에 출력하거나 다른 후속 작업을 수행합니다.
        cv2.imshow("YOLOv5 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 오디오 분류를 수행하는 함수
# 오디오 분류를 수행하는 함수
def audio_classification():
    while True:
        if ser.inWaiting() > 0:
            arduino_data = ser.readline().decode('utf-8').strip()
            if arduino_data == "Object Detected":
                print("Object detected, recording started")

                stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK,
                                input_device_index=1)

                print('start recording')

                frames = []
                seconds = 3
                for i in range(0, int(RATE / CHUNK * seconds)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                print('record stopped')

                stream.stop_stream()
                stream.close()

                recorded_file_path = 'C:/Users/hsw03/PycharmProjects/capstonedata_recorded_audio/output.wav'

                wf = wave.open(recorded_file_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                # 새로운 오디오 파일 불러오기
                new_audio_file = 'C:/Users/hsw03/PycharmProjects/capstonedata_recorded_audio/output.wav'
                test_audio = []
                y, sr = librosa.load(new_audio_file, sr=None, duration=3.0)
                test_audio.append(y)

                # 녹음 파일을 모델의 입력 형식에 맞게 변환
                dec_mfcc = []

                for y in test_audio:
                    dec = librosa.feature.mfcc(y=y, sr=sr, hop_length=940)
                    dec_mfcc.append(dec)

                mfcc = np.array(dec_mfcc, np.float32)
                mfcc = np.expand_dims(mfcc, axis=-1)

                # 모델에 적용하여 예측 수행, 결과 예측
                predictions = model.predict(mfcc)
                predicted_class = np.argmax(predictions, axis=1)

                if predicted_class == 0:
                    print("Audio_Predicted: plastic")
                    ser.write(b'plastic')  # 아두이노로 'plastic' 문자열을 보냄.
                elif predicted_class == 1:
                    print("Audio_Predicted: metal")
                    ser.write(b'metal')
                elif predicted_class == 2:
                    print("Audio_Predicted: glass")
                    ser.write(b'glass')
                else:
                    print("Audio_Predicted: Unknown")

    ser.close()



if __name__ == "__main__":


    # 오디오 분류 스레드 생성
    audio_classification_thread = threading.Thread(target=audio_classification)
    audio_classification_thread.daemon = True

    # 객체 감지 스레드 생성
    object_detection_thread = threading.Thread(target=object_detection)
    object_detection_thread.daemon = True  # 메인 스레드가 종료되면 자식 스레드도 함께 종료# 객체 감지 스레드 생성

    # 스레드 시작
    audio_classification_thread.start()
    object_detection_thread.start()


    # 스레드 종료 대기
    object_detection_thread.join()
    audio_classification_thread.join()