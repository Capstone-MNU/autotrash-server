from flask import Flask, request
import cv2
import numpy as np
from datetime import datetime
import torch
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)

from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:0000@localhost/autotrash'
db = SQLAlchemy(app)

class TrashPred(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cls = db.Column(db.String(255), nullable=False)
    conf = db.Column(db.Float, nullable=False)
    pred_date = db.Column(db.Time)

from ultralytics import YOLO
model = YOLO('best.pt')
model = model.cuda()


@app.route('/detect', methods=['POST'])
def detect_objects():
    echo = ""
    image_file = request.files['image']
    
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('capture/image.jpg', image)

    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
    image = image.cuda()
    
    result = model(image, max_det=1)
    result = result[0]
    if result.__len__() > 0: 
        cv2.imwrite('results/result.jpg', result.plot())
        box = result.boxes
        cls = result.names[box.cls.item()]
        conf = box.conf.item()
        pred_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save(cls, conf, pred_time)
        echo = cls

    return echo


def save(cls, conf, pred_time):
    record = TrashPred(cls=cls, conf=conf, pred_date=pred_time)
    db.session.add(record)
    db.session.commit()
    print("Save Object Detection Result to DB")


@app.route('/', methods=['GET'])
def test():
    print("test...")
    return "hello world!"


if __name__ == '__main__':
    app.run(debug=True)
