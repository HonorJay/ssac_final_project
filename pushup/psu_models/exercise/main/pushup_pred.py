import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import json
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

from pushup.psu_models.exercise.main.config import cfg
from pushup.psu_models.exercise.main.model import get_model
from pushup.psu_models.exercise.main.SNUEngine import SNUModel
from django.conf import settings
from pushup.psu_models.exercise.common.utils.preprocessing import load_video, augmentation, generate_patch_image, process_pose
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fitness.settings")



class PushupPred:
    def __init__(self, img_path, pose_path):
        cudnn.benchmark = True
        self.img_path = img_path
        self.pose_path = pose_path
        with open(osp.join(os.getcwd(),'pushup/psu_models/exercise/main', 'exercise_dict_body.json')) as f: # moon: exercise_dict_body 위치
            self.exer_dict = json.load(f)
        self.exer_num = len(self.exer_dict) 
        self.joint_num = 24

    def exer_pred(self, divide_list):
        cfg.set_args('0', 'exer', -1) 
        model_path = osp.join(settings.MODELS_DIR, 'exercise','exer_team2', 'snapshot_4.pth.tar')  
        model = SNUModel(model_path, self.exer_num, self.joint_num)
        exer_out = model.run(self.img_path, self.pose_path, divide_list) 
        exer_idx = np.argmax(exer_out)
        for k in self.exer_dict.keys():
            if self.exer_dict[k]['exercise_idx'] == exer_idx:
                exer_name = k
                break
        exer_out = dict()
        exer_out['exer_name'] = exer_name
        exer_out['exer_idx'] = exer_idx
        return exer_out

    def attr_pred(self, exer_name, exer_idx, seg_list):
        cfg.set_args('0', 'attr', exer_idx)
        attr_num = len(self.exer_dict[exer_name]['attr_name'])
        model_path = osp.join(settings.MODELS_DIR, 'exercise/attr_team2/' + str(exer_idx) + '/snapshot_4.pth.tar')  
        model = SNUModel(model_path, attr_num, self.joint_num)
        # seg_list: 푸시업 횟수별로 img idx를 나눠놓은 리스트; [[0,1,2,3,...],[32,33,34,..],...] 
        attr_out_list = []
        for seg in seg_list:
            attr_out = model.run(self.img_path, self.pose_path, seg)
            temp_list = []
            for i in range(len(attr_out)):
                # print(self.exer_dict[exer_name]['attr_name'][i] + ': ' + str(attr_out[i] > 0.5))
                temp_list.append(attr_out[i] > 0.5)
            attr_out_list.append(temp_list)

        return attr_out_list