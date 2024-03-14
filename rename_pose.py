import os
import open3d as o3d
from cam import *
import time
import re
import copy
from tqdm import tqdm
import time
import sys
import random
import math
import pdb
from natsort import natsorted

if __name__ == '__main__':
    model_dir = './20240306_3/sparse/0'
    cam_all = get_camera(model_dir)
    frame_num = 53
    frame_name = f'frame_{frame_num}_'
    cams = []
    for cam in cam_all:
        if frame_name in cam.name:
            cams.append(cam)
    cams = camera_group(cams,center_index='7',special_index='1')
    # cams = camera_group(get_camera(model_dir))
    pcd_dir = os.path.join(model_dir,'fused-crop.ply')
    pcd = o3d.io.read_point_cloud(pcd_dir)
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.1)
    cam_set = camera_set([cams])

    sample_root = './20240306-samples-new'

    samples = [os.path.join(sample_root, i) for i in os.listdir(sample_root)]
    samples = natsorted(samples)
    flag = 0
    all_list = []
    for sample_current in samples:
        print(sample_current)
        sample_list = os.listdir(sample_current)
        sample_list = natsorted(sample_list)
        print(sample_list)
        Rt_all = []
        
        for index in range(len(sample_list)):
            step_c = sample_list[index]
            #step_n = sample_list[index+1]

            test_rt_path = os.path.join(sample_current,step_c,'RT')
            R_dict = {}
            T_dict = {}
            for pose in os.listdir(test_rt_path):
                for iii in range(13):
                    if '_R' in pose and (f'camera_{iii}.' in pose or f'camera_{iii}_' in pose):
                        R = np.load(os.path.join(test_rt_path,pose))
                        np.save(os.path.join(test_rt_path,f'camera_{iii}_R.npy'),R)
                    if '_T' in pose and (f'camera_{iii}.' in pose or f'camera_{iii}_' in pose):
                        T = np.load(os.path.join(test_rt_path,pose))
                        np.save(os.path.join(test_rt_path,f'camera_{iii}_T.npy'),T)

            
