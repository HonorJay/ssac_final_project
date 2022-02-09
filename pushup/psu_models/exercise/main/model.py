import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.tsm.tsm_resnet import ResNetBackbone
from nets.module import Pose2Feat, Aggregator, Classifier
from nets.loss import CELoss, BCELoss
from pushup.psu_models.exercise.main.config import cfg

class Model(nn.Module):
    def __init__(self, img_backbone, pose_backbone, pose2feat, aggregator, classifier, class_num, joint_num):
        super(Model, self).__init__()
        self.img_backbone = img_backbone
        self.pose_backbone = pose_backbone
        self.pose2feat = pose2feat
        self.aggregator = aggregator
        self.classifier = classifier
        self.ce_loss = CELoss()
        self.bce_loss = BCELoss()

        self.class_num = class_num
        self.joint_num = joint_num
  
    def render_gaussian_heatmap(self, pose_coord):
        x = torch.arange(cfg.input_hm_shape[1])
        y = torch.arange(cfg.input_hm_shape[0])
        yy,xx = torch.meshgrid(y,x)
        xx = xx[None,None,:,:].cuda().float(); yy = yy[None,None,:,:].cuda().float();
        
        x = pose_coord[:,:,0,None,None]; y = pose_coord[:,:,1,None,None]; 
        heatmap = torch.exp(-(((xx-x)/cfg.hm_sigma)**2)/2 -(((yy-y)/cfg.hm_sigma)**2)/2) 
        heatmap[heatmap > 1] = 1 # threshold up to 1
        return heatmap
    
    def forward(self, inputs, targets, meta_info, mode):
        input_video = inputs['video'] # batch_size, frame_num, 3, cfg.input_img_shape[0], cfg.input_img_shape[1]
        batch_size, video_frame_num = input_video.shape[:2]
        input_video = input_video.view(batch_size*video_frame_num, 3, cfg.input_img_shape[0], cfg.input_img_shape[1])

        pose_coords = inputs['pose_coords'] # batch_size, frame_num, joint_num, 2
        batch_size, pose_frame_num = pose_coords.shape[:2]
        pose_coords = pose_coords.view(batch_size*pose_frame_num, self.joint_num, 2)
        input_pose_hm = self.render_gaussian_heatmap(pose_coords) # batch_size*pose_frame_num, self.joint_num, cfg.input_hm_shape[0], cfg.input_hm_shape[1]

        """
        # for debug
        import cv2
        import numpy as np
        import random
        _img = input_video.detach().cpu().numpy()[0] * 255
        _img = _img.transpose(1,2,0)[:,:,::-1]
        filename = random.randint(1,500)
        cv2.imwrite(str(filename) + '.jpg', _img)
        """
        
        video_feat = self.img_backbone(input_video, skip_early=False)
        pose_feat = self.pose2feat(input_pose_hm)
        pose_feat = self.pose_backbone(pose_feat, skip_early=True)
        video_pose_feat = self.aggregator(video_feat, pose_feat)
        action_out = self.classifier(video_pose_feat)
        action_out = action_out.view(batch_size, video_frame_num, -1).mean(1)
        
        if cfg.stage == 'attr':
            action_out = torch.sigmoid(action_out)
            
        if mode == 'train':
            # loss functions
            loss = {}
            if cfg.stage == 'exer':
                loss['exer'] = self.ce_loss(action_out, targets['exer_label'])
            elif cfg.stage == 'attr':
                loss['attr'] = self.bce_loss(action_out, targets['attr_label'])
            return loss

        else:
            # test output
            out = {}
            if cfg.stage == 'exer':
                out['exer'] = F.softmax(action_out,1)
            elif cfg.stage == 'attr':
                out['attr'] = action_out
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(class_num, joint_num, mode):
    img_backbone = ResNetBackbone(cfg.img_resnet_type, cfg.frame_per_seg)
    pose_backbone = ResNetBackbone(cfg.pose_resnet_type, cfg.frame_per_seg)
    pose2feat = Pose2Feat(joint_num)
    aggregator = Aggregator()
    classifier = Classifier(class_num)

    if mode == 'train':
        img_backbone.init_weights()
        pose_backbone.init_weights()
        pose2feat.apply(init_weights)
        aggregator.apply(init_weights)
        classifier.apply(init_weights)
   
    model = Model(img_backbone, pose_backbone, pose2feat, aggregator, classifier, class_num, joint_num)
    return model

