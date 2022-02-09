import torch
import torch.nn as nn
from torch.nn import functional as F
from pushup.psu_models.exercise.main.config import cfg
from pushup.psu_models.exercise.common.nets.layer import make_conv_layers, make_linear_layers

class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([self.joint_num,64])

    def forward(self, pose_heatmap):
        pose_feat = pose_heatmap
        pose_feat = self.conv(pose_feat)
        return pose_feat


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()
        self.img_resnet_dim = cfg.resnet_feat_dim[cfg.img_resnet_type]
        self.pose_resnet_dim = cfg.resnet_feat_dim[cfg.pose_resnet_type]
        self.img_conv = make_conv_layers([self.img_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)
        self.pose_conv = make_conv_layers([self.pose_resnet_dim, cfg.agg_feat_dim], kernel=1, padding=0)

    def forward(self, video_feat, pose_feat):
        pose_feat = self.pose_conv(pose_feat.mean((2,3))[:,:,None,None])
        video_feat = self.img_conv(video_feat.mean((2,3))[:,:,None,None])
        feat = video_feat + pose_feat
        return feat

class Classifier(nn.Module):
    def __init__(self, class_num):
        super(Classifier, self).__init__()
        self.class_num = class_num
        self.fc = make_linear_layers([cfg.agg_feat_dim, self.class_num], relu_final=False)

    def forward(self, video_feat):
        video_feat = video_feat.mean((2,3))
        action_out = self.fc(video_feat)
        return action_out

