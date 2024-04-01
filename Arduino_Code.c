#include <Servo.h>
#include <SoftwareSerial.h>

Servo servo1; // 첫 번째 서보 모터
Servo servo2; // 두 번째 서보 모터
int servoPin1 = 9; // 첫 번째 서보 모터 핀
int servoPin2 = 11; // 두 번째 서보 모터 핀
int irSensorPin = A0;  // IR sensor analog pin


void setup() {
  servo1.attach(servoPin1); // 첫 번째 서보 모터 핀 설정
  servo2.attach(servoPin2); // 두 번째 서보 모터 핀 설정
  Serial.begin(9600); // 시리얼 통신 시작 (아두이노와 연결된 시리얼 통신)
  servo1.write(78);
  servo2.write(85);
  delay(500);
}

void loop() {

  int irValue = analogRead(irSensorPin);
  if (irValue < 500) {  // Adjust this threshold according to your IR sensor
    Serial.println("Object Detected");  // Print to the serial monitor
  } else {
    Serial.println("No Object");  // Print to the serial monitor
  }
  delay(1000);  // Delay for stability

  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // 블루투스 모듈을 통해 데이터 수신
    // 예상 데이터 형식: "Class: Plastic, Confidence: 0.85"
    
    // 데이터에서 클래스와 신뢰도 추출
    String objectClass = "";
    float confidence = 0.0;
    
    if (parseData(data, objectClass, confidence)) {
      Serial.println("Received data: " + data);
      Serial.println("Object Class: " + objectClass);
      Serial.println("Confidence: " + String(confidence, 2)); // 소수점 둘째 자리까지 표시

      if (confidence >= 0.7) {
        if (objectClass == "Glass") {
          // Plastic 클래스일 때 서보 모터 각도 변경
          servo1.write(130); // 130도(왼쪽)로 변경
          delay(3000); // 각도 변경 후 일정 시간 유지 (1초)
          servo1.write(78); // 초기 위치로 복귀
        } else if (objectClass == "Can" || objectClass == "Plastic") {
          // Can 클래스 또는 Glass 클래스일 때 서보 모터 각도 변경
          servo1.write(20); // 20도(오른쪽)로 변경
          delay(3000); // 각도 변경 후 일정 시간 유지 (1초)
          servo1.write(78); // 초기 위치로 복귀
        } else {
          // 다른 클래스일 때 서보 모터는 아무 동작도 하지 않음
          servo1.write(78); // 초기 위치로 복귀
        }

        delay(1000);

        if (objectClass == "Can" && confidence >= 0.7) {
        // Plastic 클래스이고, Confidence가 0.8 이상일 때 서보 모터 각도 변경
        servo2.write(130); // 130도(왼쪽)로 변경
        delay(3000); // 각도 변경 후 일정 시간 유지 (1초)
        servo2.write(85); // 초기 위치로 복귀
      } else if (objectClass == "Plastic" && confidence >= 0.7) {
        // Can 클래스 또는 Glass 클래스이고, Confidence가 0.8 이상일 때 서보 모터 각도 변경
        servo2.write(20); // 20도(오른쪽)로 변경
        delay(3000); // 각도 변경 후 일정 시간 유지 (1초)
        servo2.write(85); // 초기 위치로 복귀
      } else {
        // 위 조건을 만족하지 않을 때 서보 모터는 아무 동작도 하지 않음
        servo2.write(85); // 초기 위치로 복귀
      }

      } else {
        // confidence가 0.7 미만인 경우, 두 번째 코드 실행
        String inputString = Serial.readString();
   inputString.trim(); // 앞뒤 공백 제거

        if (inputString.equals("glass")) {
          
          servo1.write(130);
          delay(2500); // 각도 변경 후 일정 시간 유지
          servo1.write(78); // 초기 위치로 복귀

         } else if (inputString.equals("metal")) {
        
        servo1.write(20);
        delay(3000);
        servo1.write(78);
        delay(1000);
       
        servo2.write(130);
        delay(3000);
        servo2.write(85);
        } else if (inputString.equals("plastic")) {
        
        servo1.write(20);
        delay(3000);
        servo1.write(78);
        delay(1000);
       
        servo2.write(20);
        delay(3000);
        servo2.write(85);
        }
      }
    }
  }
}

bool parseData(String data, String &objectClass, float &confidence) {
  int classIndex = data.indexOf("Class: ");
  int confidenceIndex = data.indexOf("Confidence: ");
  
  if (classIndex != -1 && confidenceIndex != -1) {
    // 데이터에서 클래스와 신뢰도 추출
    objectClass = data.substring(classIndex + 7, confidenceIndex - 2);
    confidence = data.substring(confidenceIndex + 12).toFloat();
    return true;
  }
  return false;
}
