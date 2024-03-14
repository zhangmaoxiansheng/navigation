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

class navigator():
    def __init__(self,cams,scene,step_size=0.2,rotate_step_size=0.02,visible=True,txt = 'record.txt',save_dir='./capture_test'):
        # if os.path.exists(txt):
        #     os.system(f'rm {txt}')
        if hasattr(cams,'center_group'):
            key_list = ['W','S','A','D','Q','E','U','O','J','L','I','K','C','N','B','V','P']
            self.cam_group = cams.center_group
            self.cam_set = cams
        else:
            key_list = ['W','S','A','D','Q','E','U','O','J','L','I','K','C']
            self.cam_group = cams
            self.cam_set = None
        self.center_cam = self.cam_group.center_cam

        width = self.center_cam.width
        height = self.center_cam.height
        self.txt = txt

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height,visible=visible)
        self.scene = scene
        
        self.vis.add_geometry(self.scene)
            
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.center_cam.o3d_cam,True)

        ###############################################
        self.step_size = step_size
        self.rotate_step_size = rotate_step_size
        self.save_dir = save_dir
        callback_dict = {}
        self.key_list = key_list
        for key in self.key_list:
            callback_dict[key]=navi_callbacks(cams,save_dir=save_dir,step_size=self.step_size,rotata_step_size=self.rotate_step_size, key=ord(key))
            self.vis.register_key_callback(ord(key),callback_dict[key].navigation_callback)
        self.cams = cams
    
    def set_step(self,step_size,rotata_step_size,callbacks_init=False):
        self.step_size = step_size
        self.rotate_step_size = rotata_step_size
        if callbacks_init:
            callback_dict = {}
            for key in self.key_list:
                callback_dict[key]=navi_callbacks(cams,save_dir=self.save_dir,step_size=self.step_size,rotata_step_size=self.rotate_step_size, key=ord(key))
                self.vis.register_key_callback(ord(key),callback_dict[key].navigation_callback)

    
    def reset_vis(self,step = 0.2, rotate_step_size=0.02,global_view = False, txt=None,visible=True,scene=None):
        self.set_step(step, rotate_step_size)
        width = self.center_cam.width
        height = self.center_cam.height
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height,visible=visible)
        # self.scene[0] = self.scene[0].sample_points_uniformly(100000)
        if scene is None:
            self.vis.add_geometry(self.scene)
        else:
            self.vis.add_geometry(scene)

        # if global_view:
        #     for p in self.cam_set.activate_points:
        #         self.vis.add_geometry(p)
        #     # self.vis.poll_events()
        #     # self.vis.update_renderer()
        self.cam_set.get_current_view_frustums()


        callback_dict = {}
        if txt is not None:
            self.txt = txt
        for key in self.key_list:
            callback_dict[key]=navi_callbacks(self.cams,save_dir=self.save_dir,step_size=self.step_size,rotata_step_size=self.rotate_step_size, key=ord(key))
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

    def replay(self,keys = None,capture=False, visible=True,global_view=True,pcds = [],delay=0,only_move = False):
        self.global_step=0
        os.makedirs(self.save_dir,exist_ok=True)
        vis = self.vis
        

        if keys is None:
            with open(self.txt,'r') as f:
                lines = f.readlines()
            keys = lines[0].split()

        if only_move:
            move_key = 'QWEASDJKLUIO'
            visible = False
            global_view = False
            capture = False
            keys = [k for k in keys if k in move_key]

        if visible and global_view:
            #按下P键保存pose
            print('press P to save pose, esc to continue')
            self.vis.run()
            self.vis.destroy_window()
        if not only_move:
            if pcds:
                self.reset_vis(global_view=global_view,visible=visible,scene=pcds[0])
            else:
                self.reset_vis(global_view=global_view,visible=visible)
        idx = 0
        deform_keys = [*range(len(self.cam_set))]
        #deform_keys_str = [str(k) for k in deform_keys]
        for key in keys:
            idx = idx + 1
            try:
                key = int(key)
            except:
                key = str(key)
            self.key = key
            print(key)
            if self.key in deform_keys:
                self.cam_set.activate_index = int(key)
                self.cam_set.update_activate_group()
                self.cam_group = self.cam_set.activate_group
                self.cam = self.cam_group.center_cam
            if self.key == ord('N') or self.key == 'N':
                self.cam_set.activate_index += 1
                self.cam_set.update_activate_group()
                self.cam_group = self.cam_set.activate_group
                self.cam = self.cam_group.center_cam
                    
            if self.key == ord('B') or self.key == 'B':
                self.cam_set.activate_index -= 1
                self.cam_set.update_activate_group()
                self.cam_group = self.cam_set.activate_group
                self.cam = self.cam_group.center_cam
            
            if self.key == ord('V') or self.key == 'V':
                self.cam_set.activate_index = 0
                for p in self.cam_set.activate_points:
                    self.vis.remove_geometry(p)
                for f in self.cam_set.activate_frustrums:
                    self.vis.remove_geometry(f)
                self.cam_set.update_activate_group()
                self.cam_group = self.cam_set.activate_group
                self.cam = self.cam_group.center_cam
                for p in self.cam_set.activate_points:
                    self.vis.add_geometry(p)
                for f in self.cam_set.activate_frustrums:
                    self.vis.add_geometry(f)
                if hasattr(self.cam_set,'vis_param'):
                    self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_set.vis_param)
                
            if self.key == ord('C') and capture:
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
                    os.makedirs(save_dir_c,exist_ok=True)
                    plt.imsave(os.path.join(save_dir_c, f"{self.global_step}_sim_image_{name}.png"), color)
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
                
                for cam in self.cam_group:
                    save_dir_c = os.path.join(self.save_dir, f"{self.global_step}",'RT')
                    os.makedirs(save_dir_c,exist_ok=True)
                    np.save(os.path.join(save_dir_c, f"{self.global_step}_sim_R_{name}.npy"), cam.R)
                    np.save(os.path.join(save_dir_c, f"{self.global_step}_sim_T_{name}.npy"), cam.T)
                self.global_step += 1
                print('capture finish!')
                
            param,extrinsic = self.navigation_callback_init()
            if self.key == ord('W') or self.key == 'W':
                extrinsic[0:3, 3] -= (self.step_size * self.forward_vector)
            if self.key == ord('S') or self.key =='S':
                extrinsic[0:3, 3] += (self.step_size * self.forward_vector)
            if self.key == ord('A') or self.key == 'A':
                extrinsic[0:3, 3] += (self.step_size * self.right_vector)
            if self.key == ord('D') or self.key == 'D':
                extrinsic[0:3, 3] -= (self.step_size * self.right_vector)
            if self.key == ord('Q') or self.key == 'Q':
                extrinsic[0:3, 3] += (self.step_size * self.up_vector)
            if self.key == ord('E') or self.key == 'E':
                extrinsic[0:3, 3] -= (self.step_size * self.up_vector)
            if self.key == ord('U') or self.key == 'U':
                rotation_matrix = np.array([[np.cos(self.rotate_step_size), -np.sin(self.rotate_step_size), 0],[np.sin(self.rotate_step_size), np.cos(self.rotate_step_size), 0],[0, 0, 1]])
                extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
                extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
            if self.key == ord('O') or self.key == 'O':
                rotation_matrix = np.array([[np.cos(-self.rotate_step_size), -np.sin(-self.rotate_step_size), 0],[np.sin(-self.rotate_step_size), np.cos(-self.rotate_step_size), 0],[0, 0, 1]])
                extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
                extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
            if self.key == ord('J') or self.key == 'J':
            # 绕Y轴旋转
                rotation_matrix = np.array([[np.cos(self.rotate_step_size), 0, np.sin(self.rotate_step_size)],[0, 1, 0],[-np.sin(self.rotate_step_size), 0, np.cos(self.rotate_step_size)]])
                extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
                extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
            if self.key == ord('L') or self.key == 'L':
                # 绕Y轴逆时针旋转
                rotation_matrix = np.array([[np.cos(-self.rotate_step_size), 0, np.sin(-self.rotate_step_size)],[0, 1, 0],[-np.sin(-self.rotate_step_size), 0, np.cos(-self.rotate_step_size)]])
                extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
                extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
            if self.key == ord('K') or self.key == 'K':
                # 绕X轴旋转
                rotation_matrix = np.array([[1, 0, 0],[0, np.cos(self.rotate_step_size), -np.sin(self.rotate_step_size)],[0, np.sin(self.rotate_step_size), np.cos(self.rotate_step_size)]])
                extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
                extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
            if self.key == ord('I') or self.key == 'I':
                # 绕X轴逆时针旋转
                rotation_matrix = np.array([[1, 0, 0],[0, np.cos(-self.rotate_step_size), -np.sin(-self.rotate_step_size)],[0, np.sin(-self.rotate_step_size), np.cos(-self.rotate_step_size)]])
                extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
                extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
            #print(self.key)
            print(self.step_size)
            param.extrinsic = extrinsic
            self.cam_group.reset_pose_all(extrinsic)
            self.cam_set.reset_pose_all(extrinsic)
            if only_move:
                continue
            if pcds:
                self.vis.remove_geometry(pcds[idx-1])
                self.vis.add_geometry(pcds[idx])

            if global_view and hasattr(self.cam_set,'vis_param'):
                # if idx > -1:
                for p in self.cam_set.activate_points:
                    self.vis.remove_geometry(p)
                for f in self.cam_set.activate_frustrums:
                    self.vis.remove_geometry(f)

                self.cam_set.get_current_view_frustums()
                
                for p in self.cam_set.activate_points:
                    self.vis.add_geometry(p)
                for f in self.cam_set.activate_frustrums:
                    self.vis.add_geometry(f)
                    
                self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam_set.vis_param)
            else:
                self.vis.get_view_control().convert_from_pinhole_camera_parameters(param)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            if delay > 0:
                time.sleep(delay)
        if not only_move:
            self.vis.destroy_window()
        return False
        
        

class navi_callbacks():
    def __init__(self,cams,save_dir='./capture_test',step_size=0.2,rotata_step_size=0.02,key=ord('W'),txt='record.txt') -> None:
        if hasattr(cams,'center_group'):
            self.cam_group = cams.center_group
            self.cam_set = cams
        else:
            self.cam_group = cams
            self.cam_set = None
        self.cam = self.cam_group.center_cam

        self.txt = txt
        self.key = key
        self.step_size = step_size
        self.rotate_step_size = rotata_step_size
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
        #print(extrinsic)
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
        if self.key == ord('W'):
            extrinsic[0:3, 3] -= (self.step_size * self.forward_vector)
        if self.key == ord('S'):
            extrinsic[0:3, 3] += (self.step_size * self.forward_vector)
        if self.key == ord('A'):
            extrinsic[0:3, 3] += (self.step_size * self.right_vector)
        if self.key == ord('D'):
            extrinsic[0:3, 3] -= (self.step_size * self.right_vector)
        if self.key == ord('Q'):
            extrinsic[0:3, 3] += (self.step_size * self.up_vector)
        if self.key == ord('E'):
            extrinsic[0:3, 3] -= (self.step_size * self.up_vector)
        if self.key == ord('U'):
            rotation_matrix = np.array([[np.cos(self.rotate_step_size), -np.sin(self.rotate_step_size), 0],[np.sin(self.rotate_step_size), np.cos(self.rotate_step_size), 0],[0, 0, 1]])
            extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
            extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
        if self.key == ord('O'):
            rotation_matrix = np.array([[np.cos(-self.rotate_step_size), -np.sin(-self.rotate_step_size), 0],[np.sin(-self.rotate_step_size), np.cos(-self.rotate_step_size), 0],[0, 0, 1]])
            extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
            extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
        if self.key == ord('J'):
        # 绕Y轴旋转
            rotation_matrix = np.array([[np.cos(self.rotate_step_size), 0, np.sin(self.rotate_step_size)],[0, 1, 0],[-np.sin(self.rotate_step_size), 0, np.cos(self.rotate_step_size)]])
            extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
            extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
        if self.key == ord('L'):
            # 绕Y轴逆时针旋转
            rotation_matrix = np.array([[np.cos(-self.rotate_step_size), 0, np.sin(-self.rotate_step_size)],[0, 1, 0],[-np.sin(-self.rotate_step_size), 0, np.cos(-self.rotate_step_size)]])
            extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
            extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
        if self.key == ord('K'):
            # 绕X轴旋转
            rotation_matrix = np.array([[1, 0, 0],[0, np.cos(self.rotate_step_size), -np.sin(self.rotate_step_size)],[0, np.sin(self.rotate_step_size), np.cos(self.rotate_step_size)]])
            extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
            extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
        if self.key == ord('I'):
            # 绕X轴逆时针旋转
            rotation_matrix = np.array([[1, 0, 0],[0, np.cos(-self.rotate_step_size), -np.sin(-self.rotate_step_size)],[0, np.sin(-self.rotate_step_size), np.cos(-self.rotate_step_size)]])
            extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
            extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
        
        #print(self.key)
        param.extrinsic = extrinsic
        if self.cam_set is not None:
            self.cam_set.reset_pose_all(extrinsic)
        else:
            self.cam_group.reset_pose_all(extrinsic)
        with open(self.txt,'a') as f:
            f.writelines(f'{self.key} ')

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
    
def adjust_cam(cam_set):
    for cameras in cam_set:
        # target_point = [-350.33,101.15799,-356.793213]
        target_point = [-19.972239,8.819079,-27.740328]
        target_camera_point = o3d.geometry.PointCloud()
        target_camera_point.points = o3d.utility.Vector3dVector([np.asarray(target_point)])
        target_camera_point.paint_uniform_color([0, 1, 0])
        cameras.transform_to(target_point)
        cameras.transform_to(target_point,[100,95,0])
        cameras.transform_to(target_point,[180,0,0])
        cameras.transform_to(target_point,[0,0,179.7])
        cameras.transform_to(target_point,[5,0,0])


if __name__ == '__main__':

    save_dir = './capture_test'
    recorder = os.path.join(save_dir,'record.txt')
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
    navi = navigator(cam_set,voxel,txt=recorder, save_dir=save_dir)
    #navi.replay()
    #random_init(navi)
    
    navi.run()