import threading
import time

import numpy as np
import torch
import argparse
import mmcv
from mmdet3d.apis import init_model
from mmdet3d.core.bbox import get_box_type
import matplotlib.pyplot as plt
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile

import warnings
warnings.filterwarnings("ignore")


def load_model(cfg_file):
    cfg = mmcv.Config.fromfile(cfg_file)
    # Set PTS voxel layer to non-deterministic to improve performance
    cfg.model.pts_voxel_layer['deterministic'] = False

    model = init_model(cfg, device='cuda:0')
    model = model.eval().half()  # Half precision to improve performance
    return model

def load_data(pcd_file, load_dims):
    pcd = LoadPointsFromFile('LIDAR', 4, 4)({'pts_filename':pcd_file})
    pcd_numpy = pcd['points'].tensor.numpy()
    return pcd_numpy

def inference_pipeline(pcd_numpy, data, model):
    #Preprocessing
    start = time.time()
    points_array = np.empty((pcd_numpy.shape[0], 5), dtype='float32')
    count = 0

    # Duplicating sequential loading
    #https://github.com/ICAV-C/conav_perception/blob/develop/conav_lidar_object_detection/conav_lidar_object_detection_mmdet3d/src/conav_lidar_object_detection/object_detection.py#L123

    for p in pcd_numpy:
        points_array[count, 0:4] = p  # x, y, z, intensity
        count += 1

    points_array = torch.tensor(points_array).to(device='cuda')
    # Intensity in the message is 0 to 100, but maybe the model uses 0 to 1?
    points_array[:, 3] = points_array[:, 3] / 100.
    points_array[:, 4] = 0.  # 5th element is just 0
    data['points'] = [[points_array]]
    preprocessing_time = time.time() - start


    start = time.time()
    # Inference with automatic mixed precision

    with torch.cuda.amp.autocast():
        result = model(return_loss=False, rescale=False, **data)
    
    inference_time = time.time() - start
    
    return preprocessing_time, inference_time

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = 'pointpillars', type = str)
    parser.add_argument("--channels", type=int, choices=[32, 64, 128])
    parser.add_argument("--nruns", type=int,default = 100)

    args, opts = parser.parse_known_args()
    
    if args.model == 'pointpillars':
        cfg_file = '/workspace/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py'
    else:
        raise NotImplementedError
    
    if args.channels == 32:
        pcd_file = '32c/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'
        ndims = 5
    if args.channels == 64:
        pcd_file = '64c/0000000001.bin'
        ndims = 4
    if args.channels == 128:
        pcd_file = '128c/0000000101.bin'
        ndims = 4

    pcd_numpy = load_data(pcd_file, ndims)
    model = load_model(cfg_file)
    box_type_3d, box_mode_3d = get_box_type('LiDAR')

    data = dict(img_metas=[[{'flip': False,
                          'pcd_horizontal_flip': False,
                          'pcd_vertical_flip': False,
                          'box_type_3d': box_type_3d,
                          'box_mode_3d': box_mode_3d,
                          'pcd_trans': np.array([0., 0., 0.]),
                          'pcd_scale_factor': 1.0,
                          'pcd_rotation': torch.tensor([[1., 0., 0.],
                                                        [-0., 1., 0.],
                                                        [0., 0., 1.]]),
                          'transformation_3d_flow': ['R', 'S', 'T']}]])

    #Warmup
    for i in range(50):
        _,_ = inference_pipeline(pcd_numpy, data, model)

    #Test
    preprocessing_time, inference_time = 0,0   
    for i in range(args.nruns):
        pre_t, inf_t = inference_pipeline(pcd_numpy, data, model)
        preprocessing_time += pre_t
        inference_time += inf_t
    pre_total = preprocessing_time/args.nruns
    inference_total = inference_time/args.nruns

    print("Total Points: ", pcd_numpy.shape[0])
    print("Preprocessing Time: ", pre_total)
    print("Inference Time: ", inference_total)
    print("FPS: ", 1/(pre_total + inference_total))

if __name__ == "__main__":
    main()