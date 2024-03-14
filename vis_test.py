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

class navigator():
    def __init__(self,cams,scene,recoder_list, visible=True,save_dir='./capture_test'):
        # if os.path.exists(txt):
        #     os.system(f'rm {txt}')
        if hasattr(cams,'center_group'):
            
            self.cam_group = cams.center_group
            self.cam_set = cams
        else:
            
            self.cam_group = cams
            self.cam_set = None
        
        key_list = ['W','S','C','N','B','V','P','A','D']
        self.center_cam = self.cam_group.center_cam

        width = self.center_cam.width
        height = self.center_cam.height

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height,visible=visible)
        self.scene = scene
        
        self.vis.add_geometry(self.scene)
            
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.center_cam.o3d_cam,True)

        ###############################################
        self.recoder_list = recoder_list
        self.save_dir = save_dir
        callback_dict = {}
        self.key_list = key_list
        for key in self.key_list:
            callback_dict[key]=navi_callbacks(cams,self.recoder_list,save_dir=save_dir,key=ord(key))
            self.vis.register_key_callback(ord(key),callback_dict[key].navigation_callback)
        self.cams = cams

    def reset_vis(self,recoder_list=None):
        self.set_step(step, rotate_step_size)
        width = self.center_cam.width
        height = self.center_cam.height
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height,visible=visible)
        # self.scene[0] = self.scene[0].sample_points_uniformly(100000)
        
        self.vis.add_geometry(self.scene)

        self.cam_set.get_current_view_frustums()
        
        if recoder_list is not None:
            self.recoder_list = recoder_list
        
        callback_dict = {}
        for key in self.key_list:
            callback_dict[key]=navi_callbacks(self.cams,self.recoder_list,save_dir=self.save_dir,key=ord(key))
            self.vis.register_key_callback(ord(key),callback_dict[key].navigation_callback)
        
        if global_view and hasattr(self.cam_set,'vis_param'):
            self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_set.vis_param)
            # self.vis.poll_events()
            # self.vis.update_renderer()
        else:
            self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.center_cam.o3d_cam,True)
        
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        self.vis.run()

class navi_callbacks():
    def __init__(self,cams, recoder_list, save_dir='./capture_test',key=ord('W')) -> None:
        if hasattr(cams,'center_group'):
            self.cam_group = cams.center_group
            self.cam_set = cams
        else:
            self.cam_group = cams
            self.cam_set = None
        self.cam = self.cam_group.center_cam
        self.l = recoder_list

        self.key = key
        self.save_dir = save_dir
        self.global_step = 0
        os.makedirs(self.save_dir,exist_ok=True)

    def navigation_callback_init(self):
        self.cam_group = self.cam_set.activate_group
        self.cam = self.cam_group.center_cam
        cam = self.cam
        param = o3d.camera.PinholeCameraParameters()
        fx, fy, cx, cy = cam.K
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(cam.width, cam.height, fx, fy, cam.width // 2-0.5, cam.height // 2-0.5)
        extrinsic = cam.pose
        self.forward_vector = extrinsic[:3,:3].T@extrinsic[:3, 2]
        self.right_vector = extrinsic[:3,:3].T@extrinsic[:3, 0]
        self.up_vector = extrinsic[:3,:3].T@extrinsic[:3, 1]
        return param,extrinsic

    def navigation_callback(self,vis):

        if self.key ==ord('P'):
            vis_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            #self.cam_set.save_global_vis_param(vis_param)
            self.cam_set.append_global_vis_param_list(vis_param)
            self.cam_set.vis_index = len(self.cam_set.vis_params) - 1
            self.cam_set.set_global_vis_param(self.cam_set.vis_index)
            print(f'view saved!total views {len(self.cam_set.vis_params)}')
            vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_set.vis_param)
            return False
        if self.key == ord('N'):
            self.cam_set.vis_index += 1
            self.cam_set.vis_index = self.cam_set.vis_index % len(self.cam_set.vis_params)
            self.cam_set.set_global_vis_param(self.cam_set.vis_index)
                
        if self.key == ord('B'):
            self.cam_set.vis_index -= 1
            self.cam_set.vis_index = self.cam_set.vis_index % len(self.cam_set.vis_params)
            self.cam_set.set_global_vis_param(self.cam_set.vis_index)
            
        if self.key == ord('V'):
            self.cam_set.activate_index = 0
            for p in self.cam_set.activate_points:
                vis.remove_geometry(p)
            for f in self.cam_set.activate_frustrums:
                vis.remove_geometry(f)
            self.cam_set.update_activate_group()
            self.cam_set.get_current_view_frustums()
            self.cam_group = self.cam_set.activate_group
            self.cam = self.cam_group.center_cam
            for p in self.cam_set.activate_points:
                vis.add_geometry(p)
            for f in self.cam_set.activate_frustrums:
                vis.add_geometry(f)
            if hasattr(self.cam_set,'vis_param'):
                vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_set.vis_param)
                return False
            
        if self.key == ord('C'):
            if self.cam_set is not None:
                current_cams = self.cam_set.activate_group
            else:
                current_cams = self.cam_group
            for i, cam in enumerate(current_cams.all_cameras):
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam.o3d_cam,True)
                color = vis.capture_screen_float_buffer(True)
                color = np.asarray(color)
                name = cam.name.split('/')[-1]
                save_dir_c = os.path.join(self.save_dir, f"{self.global_step}",'images')
                #print(save_dir_c)
                os.makedirs(save_dir_c,exist_ok=True)
                plt.imsave(os.path.join(save_dir_c, f"{self.global_step}_sim_image_{name}.png"), color)

                save_dir_c2 = os.path.join(self.save_dir, f"{self.global_step}",'RT')
                os.makedirs(save_dir_c2,exist_ok=True)
                np.save(os.path.join(save_dir_c2, f"{self.global_step}_sim_R_{name}.npy"), cam.R)
                np.save(os.path.join(save_dir_c2, f"{self.global_step}_sim_T_{name}.npy"), cam.T)

            if hasattr(self.cam_set,'vis_param'):
                for p in self.cam_set.activate_points:
                    vis.remove_geometry(p)
                for f in self.cam_set.activate_frustrums:
                    vis.remove_geometry(f)
                self.cam_set.get_current_view_frustums()
                
                for p in self.cam_set.activate_points:
                    vis.add_geometry(p)
                for f in self.cam_set.activate_frustrums:
                    vis.add_geometry(f)
                vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_set.vis_param)
            else:
                vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_group.center_cam.o3d_cam,True)
            
                    
            self.global_step += 1
            print('capture finish!')    
            with open(self.txt,'a') as f:
                f.writelines(f'{self.key} ')
            return False
        param,extrinsic = self.navigation_callback_init()

        if self.key == ord('D'):
            group = self.l.next()
            rt = self.l.c_elem.next()
            extrinsic[:3, :3] = rt['R']
            extrinsic[:3, 3] = rt['t']
            
        if self.key == ord('A'):
            group = self.l.last()
            rt = self.l.c_elem.last()
            extrinsic[:3, :3] = rt['R']
            extrinsic[:3, 3] = rt['t']
            

        if self.key == ord('W'):
            rt = self.l.c_elem.next()
            extrinsic[:3, :3] = rt['R']
            extrinsic[:3, 3] = rt['t']
            
        if self.key == ord('S'):
            rt = self.l.c_elem.last()
            extrinsic[:3, :3] = rt['R']
            extrinsic[:3, 3] = rt['t']
        
        #print(self.key)
        param.extrinsic = extrinsic
        if self.cam_set is not None:
            self.cam_set.reset_pose_all(extrinsic)
        else:
            self.cam_group.reset_pose_all(extrinsic)

        if hasattr(self.cam_set,'vis_param'):
            # if idx > -1:
            for p in self.cam_set.activate_points:
                vis.remove_geometry(p)
            for f in self.cam_set.activate_frustrums:
                vis.remove_geometry(f)
            self.cam_set.get_current_view_frustums()
            
            for p in self.cam_set.activate_points:
                vis.add_geometry(p)

            for f in self.cam_set.activate_frustrums:
                vis.add_geometry(f)
                
            vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_set.vis_param)
        else:
            vis.get_view_control().convert_from_pinhole_camera_parameters(param)
        return False

class recoder_list():
    def __init__(self,in_list):
        self.l = in_list
        self.ind = -1
        self.c_elem = in_list[0]
    def next(self):
        self.ind += 1
        self.ind = self.ind % len(self.l)
        print(self.ind)
        self.c_elem = self.l[self.ind]
        #print(self.l[self.ind])
        
        return self.l[self.ind]
    def last(self):
        self.ind -= 1
        self.ind = self.ind % len(self.l)
        return self.l[self.ind]
    def reset(self,in_list):
        self.l = in_list
        self.ind = -1
    def __len__(self):
        return len(self.l)

    def __getitem__(self,index):
        return self.l[index]

if __name__ == '__main__':
    model_dir = './reconstruction/20240306_3/sparse/0'
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

    sample_root = './samples/20240306-samples-mask'

    samples = [os.path.join(sample_root, i) for i in os.listdir(sample_root)]
    samples = natsorted(samples)
    samples = samples[-100:]
    flag = 0
    all_list = []
    for sample_current in samples:
        print(sample_current)
        sample_list = os.listdir(sample_current)
        sample_list = natsorted(sample_list)
        print(sample_list)
        Rt_all = []
        
        for index in range(len(sample_list) - 1):
            step_c = sample_list[index]
            step_n = sample_list[index+1]

            test_rt_path = os.path.join(sample_current,step_c,'test_rt_l1')
            
            R = np.load(os.path.join(test_rt_path,'camera_7_R.npy'))
            try:
                t = np.load(os.path.join(test_rt_path,'camera_7_t.npy'))
            except:
                t = np.load(os.path.join(test_rt_path,'camera_7_T.npy'))
            rt = {'R':R, 't':t}
            Rt_all.append(rt)
        
        Rt_all = recoder_list(Rt_all)
        all_list.append(Rt_all)

all_list = recoder_list(all_list)

navi = navigator(cam_set,voxel,all_list)       
navi.run()