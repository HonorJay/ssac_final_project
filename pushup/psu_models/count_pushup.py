# moon
from pushup.psu_models.exercise.main.pushup_pred import PushupPred
from pushup.psu_models.exercise.main.counting import split_count, divide_exer
from pushup.psu_models.keypoint.keypoint_generator import KeyPointGenerator
from pathlib import Path
import os
import cv2
import imageio
import numpy as np

class CountPushUp:
    """
    푸시업 개수 및 자세 추론의 일련의 과정을 메소드로 갖고 있는 클래스

    1. __init__: 사용자가 보낸 비디오 파일, 이미지와 키포인트를 저장할 디렉토리를 저장.
    2. video_framing(): 사용자가 보낸 비디오 파일을 frame별로 이미지로 img_path에 저장. 
    3. keypoint(): img_path에 저장된 frame별로 나눠진 이미지로 24개의 keypoint 좌표를 json으로 반환하여 pose_path에 저장.
    4. is_pushup(): 푸시업 여부를 추론. 
    5. split_pushup(): 푸시업 횟수별로 나눈 이미지의 인덱스 리스트를 반환.
    6. count_pushup(): 푸시업 횟수별로 푸시업 상태를 추론.
    7. result(): 결과를 client에 전달하기 위해 json 형태로 반환.

    """
    def __init__(self, input_dic):
        self.videofile_path = input_dic['videofile_path']
        self.imgfile_path = input_dic['imgfile_path']
        self.pose_path = input_dic['pose_path']

    def video_framing(self):
        cap = cv2.VideoCapture(self.videofile_path)
        cnt = 0
        if cap.isOpened():
            while True:
                ret, img = cap.read()
                if ret:
                    img = cv2.resize(img, (1920, 1080))
                    cv2.imwrite(os.path.join(self.imgfile_path,'%04d.jpg'%cnt), img)
                    cnt+=1
                else:
                    break
                # print(cnt)
        else:
            print('can\'t open video')
        cap.release()

    def keypoint(self):
        mode = KeyPointGenerator(self.imgfile_path, self.pose_path)
        mode.run()

    def is_pushup(self):
        pushup_pred = PushupPred(self.imgfile_path, self.pose_path)
        df = f'{self.pose_path}/df.csv' # moon
        divide_list = divide_exer(df) # moon
        exer_out = pushup_pred.exer_pred(divide_list) # moon : exer_pred() -> exer_pred(divide_list)
        print('exercise name:', exer_out['exer_name'])
        return pushup_pred, exer_out

    def split_pushup(self):
        df = f'{self.pose_path}/df.csv' 
        seg_list = split_count(df)
        return seg_list

    def count_pushup(self, pushup_pred, exer_out, seg_list):
        # pushup_pred = PushupPred(self.imgfile_path, self.pose_path)
        attr_out_list = pushup_pred.attr_pred(exer_out['exer_name'], exer_out['exer_idx'], seg_list)
        print('end pushup attr pred')
        return attr_out_list

    def result(self, attr_out_list):
        attr_out_list = np.array(attr_out_list)
        sum_attr_out = np.array(attr_out_list.sum(axis=1)) # 횟수별 TRUE 개수 리스트
        response_dic = {"total": str(len(attr_out_list)), "pass": str(sum(sum_attr_out > 2))}
        return response_dic