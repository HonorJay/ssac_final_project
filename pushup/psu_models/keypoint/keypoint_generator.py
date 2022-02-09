import multiprocessing
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from easydict import EasyDict
from torch import nn, optim
from tqdm import tqdm

from . import main_effdet_train as main1
from . import main_hrnet_train as main2
from . import networks
from . import utils
from .datasets import TestDetDataset, TestKeypointDataset

from importlib import import_module
from itertools import product

class KeyPointGenerator():
    def __init__(self, img_path, output_dir):
        self.img_path = img_path # 키포인트를 추출할 이미지 경로(폴더)
        self.output_dir = output_dir # 추출한 키포인트(JSON)을 저장할 경로(폴더)

        with open("pushup/psu_models/keypoint/config/effdet-d8.yaml", "r") as f: 
            C1 = EasyDict(yaml.load(f, yaml.FullLoader))
            utils.seed_everything(C1.seed)
            self.trainer1 = main1.DetTrainer(C1, 1, None)
            C1.sizes = [(768, 512), (1536, 1024)]
            C1.rotations = [0]
            C1.vflips = [False]
            C1.hflips = [False]
            C1.gammas = [1.0]
            self.C1 = C1

        with open("pushup/psu_models/keypoint/config/hrnet-w48-train-ce-512x512.yaml", "r") as f:
            C2 = EasyDict(yaml.load(f, yaml.FullLoader))

            if C2.dataset.num_cpus < 0:
                C2.dataset.num_cpus = multiprocessing.cpu_count()
            utils.seed_everything(C2.seed, deterministic=False)
            C2.result_dir = Path(C2.result_dir)
            C2.dataset.train_dir = Path(C2.dataset.train_dir)
            self.trainer2 = main2.PoseTrainer(C2, 1, "/home/ubuntu/fitness_django/inf_src/pushup/psu_models/keypoint/networks/models/HRNet-W48-ce-512x512-plus_augment-maw-rr2.0-ReduceLROnPlateau_5.pth")
            C2.sizes = [(256, 256), (512, 512), (768, 768)]
            C2.rotations = [0]
            C2.vflips = [False]
            C2.hflips = [False]
            C2.swap_columns = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 19), (22, 23)]
            self.C2 = C2

# ROI 추출
    def _get_roi(self): 
        model1 = self.trainer1.det_model
        model1.eval()
        torch.set_grad_enabled(False)

        img_files = sorted(list(Path(self.img_path).glob("*.jpg"))) # 이미지 파일 경로에 존재하는 모든 JPG파일을 리스트로 가져옴
        if len(img_files) <= 0:
            return print("please input jpg images") # 이미지를 읽어오지 못했을 때 출력할 메세지
        
        ds = TestDetDataset(self.C1, img_files)
        L = len(ds)
        roi_info = []
        for i in range(L):
            file, img = ds[i]
            file = Path(file)
            rois, scores = [], []
            for size, rotation, vflip, hflip, gamma in product(self.C1.sizes, self.C1.rotations, self.C1.vflips, self.C1.hflips, self.C1.gammas):
                img_ = img.cuda().unsqueeze(0)
                if self.C1.dataset.input_height != size[1] and self.C1.dataset.input_width != size[0]:
                    img_ = F.interpolate(img_, size[::-1])

                pannot = model1(img_)[0]

                roi, score = None, None
                for roi_, class_id_, score_ in zip(pannot["rois"], pannot["class_ids"], pannot["scores"]):
                    if class_id_ != 0:
                        continue

                    if score_ < 0.4:
                        continue

                    if roi is None or (roi[2] - roi[0]) * (roi[3] - roi[1]) < (roi_[2] - roi_[0]) * (roi_[3] - roi_[1]):
                        roi = roi_
                        score = score_

                if roi is not None and score is not None:
                    roi, score = pannot["rois"][0], pannot["scores"][0]
                    h, w = img_.shape[2:]

                    if hflip:
                        a, b, c, d = roi.copy()
                        roi = np.array([w - c, b, w - a, d], dtype=np.float32)
                    if vflip:
                        a, b, c, d = roi.copy()
                        roi = np.array([a, h - d, c, h - b], dtype=np.float32)
                    if rotation > 0:
                        a, b, c, d = roi.copy()
                        if rotation == 1:
                            roi = np.array([h - d, a, h - b, c], dtype=np.float32)
                            h, w = w, h
                        elif rotation == 2:
                            roi = np.array([w - c, h - d, w - a, h - b], dtype=np.float32)
                        elif rotation == 3:
                            roi = np.array([b, w - c, d, w - a], dtype=np.float32)
                            h, w = w, h
                    if self.C1.dataset.input_height != size[1] and self.C1.dataset.input_width != size[0]:
                        roi[0::2] = roi[0::2] * self.C1.dataset.input_width / w
                        roi[1::2] = roi[1::2] * self.C1.dataset.input_height / h
                    
                    if roi is not None and score is not None:
                        rois.append(roi)
                        scores.append(score)

            if len(rois) < 1:
                rois = [pannot['rois'][0]]
                scores = [pannot['scores'][0]]

            rois = np.stack(rois)
            roi = np.median(rois, 0)
            roi[0::2] += self.C1.dataset.crop[0]
            roi[1::2] += self.C1.dataset.crop[1]
            offset = self.C1.dataset.crop[:2]

            roi_info.append({"roi": roi.tolist(), "offset": offset})

        return roi_info

# 키포인트 추출
    def _get_keypoint(self, info):
        model2 = self.trainer2.pose_model
        model2.eval()
        torch.set_grad_enabled(False)

        img_files = sorted(list(Path(self.img_path).glob("*.jpg")))
        info_file = info

        ds_tests = [
            TestKeypointDataset(
                img_files,
                info,
                normalize=True,
                mean=self.C2.dataset.mean,
                std=self.C2.dataset.std,
                size=size,
                rotation=rotation,
                horizontal_flip=hflip,
                vertical_flip=vflip,
                ratio_limit=self.C2.dataset.ratio_limit,
            )
            for size in self.C2.sizes
            for rotation in self.C2.rotations
            for vflip in self.C2.vflips
            for hflip in self.C2.hflips
        ]

        L = len(ds_tests[0])

        # denormalize = utils.Tensor2Image(mean=self.C2.dataset.mean, std=self.C2.dataset.std)

        outdir = Path(self.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        pkey_list = []
        for i in range(L):
            phmaps = []
            for ds_test in ds_tests:
                file, img, offset, ratio, ori_size = ds_test[i]
                file = Path(file)

                phmap = model2(img.unsqueeze(0).cuda())
                if ds_test.horizontal_flip:
                    phmap = torch.flip(phmap, (3,))
                    for a, b in self.C2.swap_columns:
                        temp = phmap[:, a].clone()
                        phmap[:, a] = phmap[:, b]
                        phmap[:, b] = temp
                if ds_test.vertical_flip:
                    phmap = torch.flip(phmap, (2,))
                    for a, b in self.C2.swap_columns:
                        temp = phmap[:, a].clone()
                        phmap[:, a] = phmap[:, b]
                        phmap[:, b] = temp
                phmap = F.interpolate(phmap, ori_size[::-1])
                if ds_test.rotation > 0:
                    phmap = torch.rot90(phmap, 4 - ds_test.rotation, (2, 3))
                phmaps.append(phmap)
            phmaps = torch.cat(phmaps)

            pkeys = utils.heatmaps2keypoints(phmaps)
            pkey = pkeys.median(0).values.cpu()
            pkey[:, 0] += offset[0]
            pkey[:, 1] += offset[1]
            pkey_list.append(pkey)


        pkey_list_ = torch.stack(pkey_list)
        images = np.array(list(map(lambda x: x.name, img_files)))
        keys_out = pkey_list_.flatten(1).numpy()
        df = np.concatenate([np.expand_dims(images, 1), keys_out], 1)

        pts_info = {}
        for i in range(len(df)):
            pts_info[df[i][0]] = [[df[i][1], df[i][2]], [df[i][3], df[i][4]], [df[i][5], df[i][6]], [df[i][7], df[i][8]], 
                            [df[i][9], df[i][10]], [df[i][11], df[i][12]], [df[i][13], df[i][14]], [df[i][15], df[i][16]], 
                            [df[i][17], df[i][18]], [df[i][19], df[i][20]], [df[i][21], df[i][22]], [df[i][23], df[i][24]], 
                            [df[i][25], df[i][26]], [df[i][27], df[i][28]], [df[i][29], df[i][30]], [df[i][31], df[i][32]],
                            [df[i][33], df[i][34]], [df[i][35], df[i][36]], [df[i][37], df[i][38]], [df[i][39], df[i][40]],
                            [df[i][41], df[i][42]], [df[i][43], df[i][44]], [df[i][45], df[i][46]], [df[i][47], df[i][48]],
                            ]
        count = 0
        for i in pts_info:
            with open(outdir / f"{count:04d}.json", "w") as f:
                json.dump(pts_info[i], f)
            count += 1         
        df_columns = ['image', 'nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x',
            'right_eye_y', 'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y',
            'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x',
            'right_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'right_elbow_x',
            'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x',
            'right_wrist_y', 'left_hip_x', 'left_hip_y', 'right_hip_x',
            'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x',
            'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x',
            'right_ankle_y', 'neck_x', 'neck_y', 'left_palm_x', 'left_palm_y',
            'right_palm_x', 'right_palm_y', 'spine2(back)_x', 'spine2(back)_y',
            'spine1(waist)_x', 'spine1(waist)_y', 'left_instep_x', 'left_instep_y',
            'right_instep_x', 'right_instep_y']    
        df = pd.DataFrame(dict(enumerate(df))).T
        df.columns = df_columns
        df.to_csv(f'{outdir}/df.csv')

    def run(self):
        roi_info = self._get_roi()
        self._get_keypoint(roi_info)