# Automatic-recycling-waste-sorter-using-deep-learning <br>
2023 캡스톤 졸업작품입니다.<br>
딥러닝을 활용한 자동 재활용 쓰레기 분류기 코드 입니다. <br>
이미지 딥러닝을 위해 YOLOv5,  음성 분석 딥러닝을 위해 MFCC 알고리즘을 사용하였습니다. <br>
개발 환경은 딥러닝 학습은 Google Colab 에서, 메인 코드 실행은 Pycharm, <br>구동부 제어를 위해 Arduino IDE를 사용하였습니다.<br>
각 코드와 최종 보고서 첨부합니다.<br> <br>


## 작품 개요

객체인식을 수행하여 YOLOv5모델을 이용해 물체의 재질을 판별하고 그와 동시에 물체가 떨어지는 음성의 데이터를 전처리하여 CNN 모델을 이용해 독립적으로 한 번 더 재질을 판별하도록 한다. 이는 두 종류의 기계학습을 실행하여 부족한 신뢰도를 보완하기 위함이다. <br>
 각각의 판별 결과를 출력하고 이 결과를 시리얼 통신으로 아두이노에 송신한다. 객체 인식의 결과를 출력하고 신뢰도가 0.7 이상이면 이 결과를 최종 결과로 판단한다. <br>
 만약 객체 인식을 통한 판별 결과의 신뢰도가 0.7 미만인 경우, 음성 데이터를 이용한 판별 결과를 최종 결과로 판단하게 된다. 최종 판별 결과가 정해지면 이 결과에 따라 아두이노에서는 2개의 서보모터를 순차적으로 제어하여 물체를 서로 다른 곳으로 분리한다. <br><br>

## 작품 구성도
![image](https://github.com/user-attachments/assets/7b9f5f8a-1424-456a-bdc1-011a4f1bb585) <br>
## 실제 동작 영상 링크
https://youtu.be/hUXsNtIZoiM <br><br>

## 작품 사진 
![작품사진](https://github.com/YuNyuk/Automatic-recycling-waste-sorter-using-deep-learning/assets/142381053/cd9a3d2a-465b-4e10-8485-f63ff06be0ad) <br>
![작품사진_판넬](https://github.com/YuNyuk/Automatic-recycling-waste-sorter-using-deep-learning/assets/142381053/30806e6f-0d4f-4fde-a679-30e635763da0)
