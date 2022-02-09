import sys
import os
import os.path as osp
import numpy as np
import cv2
import torch
import json
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import random

from pushup.psu_models.exercise.main.config import cfg
from pushup.psu_models.exercise.main.model import get_model
from pushup.psu_models.exercise.common.utils.preprocessing import load_video, get_bbox, process_bbox, augmentation, generate_patch_image, process_pose

class SNUModel:
    def __init__(self, model_path, class_num, joint_num):
        # snapshot load
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print('Load checkpoint from {}'.format(model_path))
        model = get_model(class_num, joint_num, 'test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.joint_num = joint_num
        self.model = model
    
    def run(self, img_path, pose_path, seg):
        # prepare input video
        # rgb video
        img_path_list = [osp.join(img_path, '%04d.jpg' % i) for i in seg] # moon

        # img_path에서 cfg.frame_per_seg 개를 랜덤하게 추출
        rand_img_path_list = random.sample(img_path_list, cfg.frame_per_seg) # moon
        
        start_frame_idx = 0
        video, video_frame_idxs, original_img_shape = load_video(img_path_list, start_frame_idx)
        resized_shape = video.shape[1:3] # height, width

        # pose video
        pose_coords = []
        # pose_path도 cfg.frame_per_seg 개를 랜덤하게 추출
        pose_path_list = [osp.join(pose_path, '%04d.json' % i) for i in seg] # moon

        rand_pose_path_list = random.sample(pose_path_list, cfg.frame_per_seg) # moon

        for i in range(cfg.frame_per_seg):
            with open(rand_pose_path_list[i]) as f: # moon
                pose_coord = np.array(json.load(f), dtype=np.float32).reshape(self.joint_num,2)
            pose_coords.append(pose_coord)
        pose_coords = np.stack(pose_coords) # cfg.frame_per_seg, joint_num, 2
        pose_coords[:,:,0] = pose_coords[:,:,0] / original_img_shape[1] * resized_shape[1]
        pose_coords[:,:,1] = pose_coords[:,:,1] / original_img_shape[0] * resized_shape[0]

        # augmentation
        bboxs = []
        for i in range(len(video_frame_idxs)):
            bbox = get_bbox(pose_coords[i], np.ones_like(pose_coords[i,:,0]))
            bbox = process_bbox(bbox)
            bboxs.append(bbox)
        bboxs = np.array(bboxs).reshape(-1,4) # xmin, ymin, width, height
        bboxs[:,2] += bboxs[:,0]; bboxs[:,3] += bboxs[:,1]; # xmin, ymin, xmax, ymax
        xmin = np.min(bboxs[:,0]); ymin = np.min(bboxs[:,1]);
        xmax = np.max(bboxs[:,2]); ymax = np.max(bboxs[:,3]);
        bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin], dtype=np.float32) # xmin, ymin, width, height
        video, img2aug_trans = augmentation(video, bbox)
        video = video.transpose(0,3,1,2).astype(np.float32)/255. # frame_num, channel_dim, height, width
        for i in range(len(video_frame_idxs)):
            pose_coords[i] = process_pose(pose_coords[i], img2aug_trans, self.joint_num, resized_shape)
            
        # forward
        video = torch.from_numpy(video)[None,:,:,:,:]
        pose_coords = torch.from_numpy(pose_coords)[None,:,:,:]
        inputs = {'video': video, 'pose_coords': pose_coords}
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = self.model(inputs, targets, meta_info, 'test')
        action_out = out[cfg.stage][0].detach().cpu().numpy()
        return action_out

