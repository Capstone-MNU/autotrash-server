from flask import Flask, request
import cv2
import numpy as np
from datetime import datetime
import torch
from flask_sqlalchemy import SQLAlchemy
import pymysql
pymysql.install_as_MySQLdb()



app = Flask(__name__)


from sqlalchemy import Column, Integer, String, Date, DECIMAL, Enum, ForeignKey, Time
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(255))
    password = Column(String(266))
    user_level = Column(Enum('실버', '골드', '다이아몬드'))
    eco_points = Column(Integer)
    

class TrashBin(Base):
    __tablename__ = 'trash_bin'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    state = Column(String(255))
    trash_type = Column(String(255))
    disposal_date = Column(Date)
    disposal_amount = Column(DECIMAL(10,2))

    user = relationship('Users', back_populates='trash_bins')

    
class TrashBinCapacity(Base):
    __tablename__ = 'trash_bin_capacity'

    id = Column(Integer, primary_key=True)
    trash_bin_id = Column(Integer, ForeignKey('trash_bin.id'))
    check_date = Column(Date)
    check_time = Column(Time)
    capacity = Column(DECIMAL(10,2))

    trash_bin = relationship('TrashBin', back_populates='capacities')


Users.trash_bins = relationship('TrashBin', order_by=TrashBin.id, back_populates='user')
TrashBin.capacities = relationship('TrashBinCapacity', order_by=TrashBinCapacity.id, back_populates='trash_bin')

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


@app.route('/ultwave', methods=['GET'])
def save_ult_wave():
    dist = request.args['dist']
    print(dist)
    cls = request.args['cls']
    
    # users 테이블에서 id가 2인 user 검색
    user = db.session.query(Users).get(2)
    
    # 해당 user와 연결된 trash_bin 테이블에서 trash_bin_id 검색
    for trash_bin in user.trash_bins:
        print(cls)
        if trash_bin.trash_type in cls: 
            # 해당 trash_bin_id와 연결된 모든 trash_bin_capacity 테이블의 레코드에 dist 값 저장
            for capacity in trash_bin.capacities:
                capacity.capacity = dist
            
    db.session.commit()

    return '저장 완료'


@app.route('/ecop', methods=['GET'])
def plus_ecopoint():
    
    # users 테이블에서 id가 2인 user 검색
    user = db.session.query(Users).get(2)
    
    user.eco_points += 10
    
    db.session.commit()
    return 'eco_points 증가 완료'

@app.route('/', methods=['GET'])
def test():
    print("test...")
    return "hello world!"


if __name__ == '__main__':
    app.run(debug=True)
