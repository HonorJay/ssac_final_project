import os
import os.path as osp
import sys
import numpy as np

class Config:
 
    ## experiment
    stage = 'exer' #'attr' # exer, attr
    exer_idx = '-1' #10  #iylee: 0 --> 10

    ## dataset
    dataset = 'Sleek'
    
    ## model setting
    img_resnet_type = 18 # 18, 34, 50, 101, 152
    pose_resnet_type = 18 # 18, 34, 50, 101, 152
    resnet_feat_dim = {18: 512, 34: 512, 50: 2048}
    agg_feat_dim = 512

    ## input config
    video_shape = (256,)
    input_img_shape = (224, 224)
    input_hm_shape = (56, 56)
    hm_sigma = 0.5
    frame_per_seg = 8
        
    ## training config
    lr = 5e-4
    lr_dec_epoch = [4] #[21,26]
    end_epoch = 5 #30
    lr_dec_factor = 10
    momentum = 0.9
    weight_decay = 1e-4
    train_batch_size = 8  #iylee: 8 --> 4 ?

    ## testing config
    test_batch_size = 16

    ## others
    num_thread = 16  #iylee: 40 --> 16
    gpu_ids = '0,1' #iylee: '0' --> '0,1'
    num_gpus = 2  #iylee: 1 --> 2
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    
    def set_args(self, gpu_ids, stage, exer_idx, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.stage = stage
        self.exer_idx = int(exer_idx)
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        print('Stage: ' + self.stage)

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))  #iylee: dataset path
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
