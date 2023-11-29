from flask import Flask, request
import cv2
import numpy as np
import mysql.connector

app = Flask(__name__)

# 객체 검출을 위한 YOLOv8 모델 로드
from ultralytics import YOLO
model = YOLO('best.pt')

@app.route('/detect', methods=['POST'])
def detect_objects():
    # 클라이언트로부터 이미지 파일을 받아옴
    image_file = request.files['image']
    
    # 이미지 파일을 OpenCV로 읽어옴
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # YOLOv8을 사용하여 객체 검출 수행
    result = model(image, max_det=1)
    box = result[0].boxes
    pred_cls = int(box[0].cls)
    pred_probs = result[0].probs
    
    # 검출된 객체 정보를 MySQL DB에 저장
    # MySQL DB에 접속
    db = mysql.connector.connect(
        host="112.175.185.132",
        user="rkdtjddn132",
        password="tjddn132!",
        database="rkdtjddn132"
    )
    cursor = db.cursor()
    # 커서 생성


    # 검출 결과를 MySQL DB에 저장
    # ...

    # 커밋
    db.commit()

    # 연결 종료
    db.close()

    return 'Object detection result saved in MySQL DB'







if __name__ == '__main__':
    app.run()
