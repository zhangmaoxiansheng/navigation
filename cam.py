import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

import open3d as o3d
import cv2
from bin2txt import *
import copy


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

class camera:
    def __init__(self,K,R,T,name='cam',width=1280,height=720,near_clip=0.01,far_clip=0.5, rgb = None, depth = None, cam_model='PINHOLE') -> None:
        self.K = K
        fx, fy, cx, cy = self.K
        self.K_array = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
        self.R = R
        self.T = T
        if cam_model == 'PINHOLE':
            self.width = int(2 * cx)
            self.height = int(2 * cy)
        else:
            self.width= width
            self.height = height
        self.camera_position = -R.T@T
        self.name = name
        self.frustum_vertices = []
        self.pose = np.identity(4)
        self.pose[:3,:3] = self.R
        self.pose[:3,3] = self.T 
        self.to_o3d()
        self.near_clip=near_clip
        self.far_clip=far_clip
        self.rgb = rgb
        self.depth = depth
        self.cam_model = cam_model

    def attach_rgb(self,rgb):
        self.rgb = rgb

    def attach_depth(self,depth):
        self.depth = depth

    def to_o3d(self):
        #cpu_device = o3d.core.Device("CPU:0")
        cam = o3d.camera.PinholeCameraParameters()
        fx, fy, cx, cy = self.K
        #openGL problem!

        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, self.width // 2-0.5, self.height // 2-0.5)
        extrinsic = self.pose
        cam.extrinsic = extrinsic
        self.o3d_cam = cam

    def rotate(self,R):
        self.R = R @ self.R
        self.T = R @ self.T
        self.pose[:3,:3] = self.R
        self.pose[:3,3] = self.T 
        self.camera_position = -self.R.T@self.T
        self.to_o3d()

    def transform(self,trans_mat):
        self.pose = trans_mat @ self.pose 
        self.R = self.pose[:3,:3]
        self.T = self.pose[:3,3]
        self.camera_position = -self.R.T@self.T
        self.to_o3d()
    
    def reset_pose(self,pose):
        self.pose = pose
        self.R = self.pose[:3,:3]
        self.T = self.pose[:3,3]
        self.camera_position = -self.R.T@self.T
        self.to_o3d()
    
    def adjust_scale(self, scale):
        self.T = self.T * scale
        self.pose[:3,3] = self.T
        self.camera_position = -self.R.T@self.T
        self.to_o3d()

    def get_view_frustum(self,cam_color = [1, 0, 0],line_color=[0, 0, 1], line_radius = -1):
        
        near_clip = self.near_clip
        far_clip = self.far_clip
        K = self.K
        # 相机内部参数
        fx, fy, cx, cy = K

        # 图像分辨率
        R = self.R
        T = self.T
        width = self.width
        height = self.height
        # 相机的外部参数（相机的位置和方向）
        # 这里只是一个示例，你需要提供实际的相机外部参数
        camera_position = self.camera_position  # 相机位置
        

        # 创建一个点云表示相机位置
        camera_point = o3d.geometry.PointCloud()
        camera_point.points = o3d.utility.Vector3dVector([camera_position])
        camera_point.paint_uniform_color(cam_color)  # 设置点的颜色为红色

        # 创建表示相机视锥体的线集
        self.line_set = o3d.geometry.LineSet()

        # 计算相机视锥体的八个顶点

        # 视锥体的四个角点在相机坐标系下的坐标
        top_left = np.array([(0 - cx) * near_clip / fx, (0 - cy) * near_clip / fy, near_clip])
        top_right = np.array([(width - cx) * near_clip / fx, (0 - cy) * near_clip / fy, near_clip])
        bottom_left = np.array([(0 - cx) * near_clip / fx, (height - cy) * near_clip / fy, near_clip])
        bottom_right = np.array([(width - cx) * near_clip / fx, (height - cy) * near_clip / fy, near_clip])

        # 通过相机的旋转将角点变换到世界坐标系
        vertices = [np.dot(R.T, top_left) + camera_position,
                    np.dot(R.T, top_right) + camera_position,
                    np.dot(R.T, bottom_right) + camera_position,
                    np.dot(R.T, bottom_left) + camera_position,
                    np.dot(R.T, top_left) * (far_clip / near_clip) + camera_position,
                    np.dot(R.T, top_right) * (far_clip / near_clip) + camera_position,
                    np.dot(R.T, bottom_right) * (far_clip / near_clip) + camera_position,
                    np.dot(R.T, bottom_left) * (far_clip / near_clip) + camera_position]

        # 定义相机视锥体的边
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]]

        self.line_set.points = o3d.utility.Vector3dVector(vertices)
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
        self.line_set.colors = o3d.utility.Vector3dVector(np.array([line_color for _ in range(len(lines))]))  # 设置线的颜色为蓝色

        colors = [line_color for i in range(len(lines))]
        vertices_array = np.array(vertices)
        if line_radius < 0:
            line_mesh1_geoms = self.line_set
        else:
            line_mesh1 = LineMesh(vertices_array, lines, colors, radius=line_radius)
            line_mesh1_geoms = line_mesh1.cylinder_segments
        self.frustum_vertices = vertices
        
        near_plane = [self.frustum_vertices[0], self.frustum_vertices[1], self.frustum_vertices[2], self.frustum_vertices[3]]
        far_plane = [self.frustum_vertices[4], self.frustum_vertices[5], self.frustum_vertices[6], self.frustum_vertices[7]]
        left_plane = [self.frustum_vertices[0], self.frustum_vertices[3], self.frustum_vertices[7], self.frustum_vertices[4]]
        right_plane = [self.frustum_vertices[1], self.frustum_vertices[5], self.frustum_vertices[6], self.frustum_vertices[2]]
        top_plane = [self.frustum_vertices[0], self.frustum_vertices[1], self.frustum_vertices[5], self.frustum_vertices[4]]
        bottom_plane = [self.frustum_vertices[3], self.frustum_vertices[2], self.frustum_vertices[6], self.frustum_vertices[7]]

        self.planes = [near_plane, far_plane, left_plane, right_plane, top_plane, bottom_plane]
        self.planes_bool = [True,False,True,True,False,True]
        self.planes_coeffs = []
        for plane in self.planes:
        # 计算平面方程 Ax + By + Cz + D = 0 中的 ABCD 系数
            self.planes_coeffs.append(self.plane_equation(plane))

        return camera_point, line_mesh1_geoms

    def frustum_check(self,point):
        if not self.frustum_vertices:
            self.get_view_frustum() 
        for coeff,is_positive in zip(self.planes_coeffs,self.planes_bool):   
            A,B,C,D = coeff
            distance = A * point[0] + B * point[1] + C * point[2] + D
            #print(distance)
            if (is_positive and distance < 0) or (not is_positive and distance > 0):
                return False
        return True
    
    def project_3d_2d(self,p):
        camera_point = self.R @ p + self.T
        x,y,z = camera_point
        fx, fy, cx, cy = self.K
        
        u = int((fx * x)/ z + cx)
        v = int((fy * y) / z + cy)

        #projection = self.K_array @ camera_point
        #u,v = round(projection[0] / projection[2]),round(projection[1] / projection[2])
        
        return u,v,z

    def visible_check_depth(self,p, depth_in = None):
        if depth_in is None and self.depth is None:
            print('no depth!fail!!')
            return False
        if depth_in is not None:
            depth = depth_in
        if depth_in is None and self.depth is not None:
            depth = self.depth
        # plt.imshow(depth)
        # plt.show()
        u,v,z = self.project_3d_2d(p)
        #######################################################
        if z < 0 or z > self.far_clip:  #point behind the camera or out of the far plane not visible
            return 0
        if u<0 or u>self.width-1 or v<0 or v>self.height-1:#out of range not visible
            return 0
        d = depth[v,u]

        if d == 0:#in the range but no depth   1:overlap not enough   2:the sky, can not reconstruct
            return 1
        if z > d:#occulusion not visible
            return 0
        if z < d:#empty
            return -1

    def visible_check_voxel(self,p,voxel_grid):
        in_frustum = self.frustum_check(p)
        if not in_frustum:
            return False

        aabb = voxel_grid.get_axis_aligned_bounding_box()
        aabb = aabb.get_box_points()
        aabb = np.asarray(aabb)
        x_min = np.min(aabb[:,0])
        x_max = np.max(aabb[:,0])

        y_min = np.min(aabb[:,1])
        y_max = np.max(aabb[:,1])

        z_min = np.min(aabb[:,2])
        z_max = np.max(aabb[:,2])

        min_v = np.array([x_min,y_min,z_min])
        max_v = np.array([x_max,y_max,z_max])
        output = False
        sample_density = 100
        
        if (p >= min_v).all() and (p <= max_v).all():
            line_queries = np.linspace(p, self.camera_position, sample_density)
            output_p = voxel_grid.check_if_included(o3d.utility.Vector3dVector(line_queries))
            if any(output_p):
                output = False
            else:
                output = True
        else:
            output = True
        return output

    def capture(self, scene, save_dir = None, with_depth = False, show = False, save_in_cam = False, group_id = 0):
        if save_dir is not None:
            os.makedirs(save_dir,exist_ok=True)
        width = self.center_cam.width
        height = self.center_cam.height

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height,visible=False)
        
        for obj in scene:
            vis.add_geometry(obj)
        
        vis.get_view_control().convert_from_pinhole_camera_parameters(self.o3d_cam,True)
        vis.update_renderer()
        color = vis.capture_screen_float_buffer(True)
        color = np.asarray(color)
        if save_in_cam:
            self.attach_rgb(color)
        if show:
            plt.imshow(color)
            plt.show()
        if save_dir is not None:
            name = self.name.split('/')[-1]
            plt.imsave(os.path.join(save_dir, f"{group_id}_sim_image_{name}.png"), color)

        if with_depth:
            depth = vis.capture_depth_float_buffer(True)
            depth = np.asarray(depth)
            if save_in_cam:
                self.attach_depth(depth)
            if save_dir is not None:
                np.save(os.path.join(save_dir, f"{group_id}_sim_depth_{name}.npy"), depth)
            if show:
                plt.imshow(depth)
                plt.show()
        vis.destroy_window()
        if save_dir is not None:
            print(f'images save to {save_dir}')
        return color, depth
    
    def rgbd2pcd(self):
        if self.rgb is not None and self.depth is not None:
            color_raw = o3d.geometry.Image((255*self.rgb).astype(np.uint8))
            depth_raw = o3d.geometry.Image(self.depth)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw, depth_raw,depth_scale=1, depth_trunc=400000.0, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,self.o3d_cam.intrinsic)
            self.pcd = pcd
            return True
        else:
            print('Fail! No rgb or depth!')
            return False
        
    @staticmethod
    def plane_equation(plane):
        points = np.array(plane)
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1,v2)
        A,B,C = normal
        D = -np.dot(normal, points[0])

        return A, B, C, D


class camera_group:
    def __init__(self,cameras,center_index='11',center_name=None,special_index=None) -> None:
        self.all_cameras = cameras
        self.outer_cameras = []
        for cam in cameras:
            if center_name is None:
                cam_index = cam.name.split('_')[-1]
                if  (not hasattr(self,'center_cam')) and center_index == cam_index:
                    self.center_cam = cam
                    self.center_name = cam.name
                else:
                    self.outer_cameras.append(cam)
            else:
                if (not hasattr(self,'center_cam')) and cam.name == center_name:
                    self.center_cam = cam
                    self.center_name = cam.name
                else:
                    self.outer_cameras.append(cam)
        
        self.relative_poses = {}
        for cam in self.outer_cameras:
            self.relative_poses[cam.name] = np.dot(cam.pose,np.linalg.inv(self.center_cam.pose))
        #print(self.center_cam.name)
        self.all_dict = {}
        for cam in self.all_cameras:
            self.all_dict[cam.name]=cam
        self.special_index = special_index
    def __len__(self):
        return len(self.all_cameras)

    def __getitem__(self,index):
        return self.all_cameras[index]
    def rgbd2pcd(self):
        for cam in self.all_cameras:
            flag = cam.rgbd2pcd()
            if not flag:
                break
    
    def clear_rgbd(self):
        for cam in self.all_cameras:
            cam.rgb = None
            cam.depth = None
            cam.pcd = None

    def write2txt(self,path):
        #colmap format
        #write camera
        HEADER = "# Camera list with one line of data per camera:\n" + \
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
            "# Number of cameras: {}\n".format(len(self.all_cameras))
        with open(os.path.join(path,'cameras.txt'), "w") as fid:
            fid.write(HEADER)
            for id,cam in enumerate(self.all_cameras):
                to_write = [id, cam.cam_model, cam.width, cam.height, *cam.K]
                line = " ".join([str(elem) for elem in to_write])
                fid.write(line + "\n")
        #write images(extrin)
        
        mean_observations = 0
        HEADER = "# Image list with two lines of data per image:\n" + \
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
                "# Number of images: {}, mean observations per image: {}\n".format(len(self.all_cameras), mean_observations)
    
        with open(os.path.join(path,'images.txt'), "w") as fid:
            fid.write(HEADER)
            for id,cam in enumerate(self.all_cameras):
                qvec = rotmat2qvec(cam.R)
                qvec = list(qvec)
                tvec = list(cam.T)
                image_header = [id, *qvec, *tvec, id, cam.name+'.png']
                first_line = " ".join(map(str, image_header))
                fid.write(first_line + "\n")
                #second line is useless
                fid.write(" " + "\n")


    def append_cams_list(self,cam_list):
        #assume we do not change the center cam!
        for cam in cam_list:
            self.all_cameras.append(cam)
            self.outer_cameras.append(cam)
            self.relative_poses[cam.name] = np.dot(cam.pose,np.linalg.inv(self.center_cam.pose))

    def append_cam(self,cam):
        #assume we do not change the center cam!
        self.all_cameras.append(cam)
        self.outer_cameras.append(cam)
        self.relative_poses[cam.name] = np.dot(cam.pose,np.linalg.inv(self.center_cam.pose))


    def capture(self, scene, save_dir = None, with_depth = False, show = False, save_in_cam = True, group_id = 0):
        if save_dir is not None:
            os.makedirs(save_dir,exist_ok=True)
        width = self.center_cam.width
        height = self.center_cam.height

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height,visible=False)
        
        for obj in scene:
            vis.add_geometry(obj)
        rgb_list = []
        depth_list = []
        ####very dangerous bug!!!!
        for i, cam in enumerate(self.all_cameras):
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam.o3d_cam,True)
            vis.update_renderer()
            color = vis.capture_screen_float_buffer(True)
            color = np.asarray(color)
            if save_in_cam:
                cam.attach_rgb(color)
            if show:
                plt.imshow(color)
                plt.show()
            rgb_list.append(color)
            if save_dir is not None:
                name = cam.name.split('/')[-1]
                plt.imsave(os.path.join(save_dir, f"{group_id}_sim_image_{name}.png"), color)

            if with_depth:
                depth = vis.capture_depth_float_buffer(True)
                depth = np.asarray(depth)
                if save_in_cam:
                    cam.attach_depth(depth)
                if save_dir is not None:
                    np.save(os.path.join(save_dir, f"{group_id}_sim_depth_{name}.npy"), depth)
                depth_list.append(depth)
                if show:
                    plt.imshow(depth)
                    plt.show()
        vis.destroy_window()
        if save_dir is not None:
            print(f'images save to {save_dir}')
        return rgb_list, depth_list
    
    def transform_to(self, position_B, rotation_angle = [0,0,0]):
        position_B = np.array(position_B)
        # 定义旋转角度（以度为单位）
        rotation_angle_x = rotation_angle[0]  # 绕X轴的旋转角度
        rotation_angle_y = rotation_angle[1]  # 绕Y轴的旋转角度
        rotation_angle_z = rotation_angle[2]  # 绕Z轴的旋转角度
        r = R.from_euler('xyz', [rotation_angle_x, rotation_angle_y, rotation_angle_z],True)
        rotation_matrix = r.as_matrix()

        transformation_matrix = np.eye(4)  # 创建一个4x4的单位矩阵
        transformation_matrix[:3, :3] = rotation_matrix @ self.center_cam.R  # 设置旋转部分
        transformation_matrix[:3, 3] = -rotation_matrix @ self.center_cam.R @ position_B  # 设置平移部分
        
        self.center_cam.reset_pose(transformation_matrix)
        for cam in self.outer_cameras:
            relative_pose = self.relative_poses[cam.name]
            new_pose = np.dot(relative_pose, self.center_cam.pose)
            cam.reset_pose(new_pose)
    
    def reset_pose_all(self,center_pose):
        self.center_cam.reset_pose(center_pose)
        for cam in self.outer_cameras:
            relative_pose = self.relative_poses[cam.name]
            new_pose = np.dot(relative_pose, self.center_cam.pose)
            cam.reset_pose(new_pose)
    
    def adjust_scale(self,scale):
        for cam in self.all_cameras:
            cam.adjust_scale(scale)
        for cam in self.outer_cameras:
            self.relative_poses[cam.name] = np.dot(cam.pose,np.linalg.inv(self.center_cam.pose))

    def rotate(self,R):
        for cam in self.all_cameras:
            cam.rotate(R)

    def transform(self, transform_matrix):
        for cam in self.all_cameras:
            cam.transform(transform_matrix)
    
    def get_aabb(self,scale=1):
        if not hasattr(self,'frustum_vertices'):
            cam_points, frustums = self.get_view_frustums()
        vertices = []
        for cam in self.all_cameras:
            
            vertices = vertices + cam.frustum_vertices[:4] + [y+scale*(x-y) for x,y in zip(cam.frustum_vertices[4:],cam.frustum_vertices[:4])]
    
        vertices = np.asarray(vertices)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        aabb = pcd.get_axis_aligned_bounding_box()  

        aabb.color = (0, 1, 0)
        #all_vis = [aabb,pcd] + cam_points + frustums
        #o3d.visualization.draw_geometries(all_vis)
        points = aabb.get_box_points()
        points = np.asarray(points)
        return aabb,points
    
    def get_obb(self):
        cameras = self.all_cameras
        vertices = []
        for cam in cameras:
            vertices += cam.frustum_vertices
        vertices = np.asarray(vertices)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        aabb = pcd.get_oriented_bounding_box()
        aabb.color = (0, 1, 0)
        points = aabb.get_box_points()
        points = np.asarray(points)
        return aabb,points
    
    def check_visible_overlap(self, queries, voxel_grid=None, occu_list = [],filter_thres = 10):
        cameras = self.all_cameras
        vertices = []
        for cam in cameras:
            vertices += cam.frustum_vertices
        # max_view = len(cameras)
        # filter_thres = min(filter_thres,max_view)
        vertices = np.array(vertices)
        
        x_min = np.min(vertices[:,0])
        x_max = np.max(vertices[:,0])

        y_min = np.min(vertices[:,1])
        y_max = np.max(vertices[:,1])

        z_min = np.min(vertices[:,2])
        z_max = np.max(vertices[:,2])

        min_v = np.array([x_min,y_min,z_min])
        max_v = np.array([x_max,y_max,z_max])
        output_list = []
        #occupancy is -1, empty point is > filter_thres
        for ind,p in enumerate(queries):
            if occu_list:
                if occu_list[ind]>filter_thres:
                    output_list.append(occu_list[ind])
                    continue
                elif occu_list[ind]>0:
                    output_list.append(-1)
                    continue

            if (p >= min_v).all() and (p <= max_v).all():
                count = 0
                for cam in cameras:
                    if cam.depth is None:
                        if cam.visible_check_voxel(p,voxel_grid):
                            count += 1
                    else:
                        res = cam.visible_check_depth(p)
                        if res < 0:
                            count = filter_thres + 1     #empty filtered out!
                            break
                        if res > 0:
                            count += 1
                output_list.append(count)
            else:
                output_list.append(0)
        return output_list

    def get_view_frustums(self,line_radius=-1):
        cam_points = []
        frustums = []
        for cam in self.all_cameras:
            p,frustum_line = cam.get_view_frustum(line_radius = line_radius)
            if self.special_index is not None:
                cam_index = cam.name.split('_')[-1]
                if self.special_index == cam_index:
                    p.paint_uniform_color([0,1,0])
            cam_points.append(p)
            if line_radius < 0:
                frustums.append(frustum_line)
            else:
                frustums+= frustum_line
        return cam_points, frustums
    
    def cal_neighbor_dist(self, n1='7', n2='1'):
        neighbor_points = []
        for cam in self.all_cameras:
            cam_index = cam.name.split('_')[-1]
            if n1 == cam_index or n2 == cam_index:
                neighbor_points.append(cam.camera_position)
        distance = np.linalg.norm(neighbor_points[1] - neighbor_points[0])
        return distance

    def save_camera_pos(self, file_path, cam_color = [1, 0, 0]):
        points = []
        for cam in self.all_cameras:
            points.append(cam.camera_position)
        
        camera_point = o3d.geometry.PointCloud()
        camera_point.points = o3d.utility.Vector3dVector(points)
        camera_point.paint_uniform_color(cam_color)  # 设置点的颜色为红色
        p,frustum_line = self.center_cam.get_view_frustum()

        o3d.io.write_point_cloud(os.path.join(file_path, 'cameras.ply'), camera_point)
        o3d.io.write_line_set(os.path.join(file_path, 'centercam_frustum.ply'), self.center_cam.line_set)
    
    def attach_score(self,pos,score_array):
        self.pos = pos
        self.score_array = score_array
    
class camera_set():
    #include some camera group
    def __init__(self,cam_group_list) -> None:
        #choose the first as the center
        self.center_group = cam_group_list[0]
        self.center_cam = self.center_group.center_cam
        if len(cam_group_list) > 1:
            self.other_cg_list = cam_group_list[1:]
            self.relative_poses = {}
            for ind, cg in enumerate(self.other_cg_list):
                cam = cg.center_cam
                self.relative_poses[ind+1] = np.dot(cam.pose,np.linalg.inv(self.center_cam.pose))
        else:
            self.other_cg_list = None
        self.all_cgs = cam_group_list
        self.activate_index = 0
        self.activate_group = self.center_group
        self.activate_points,self.activate_frustrums = self.activate_group.get_view_frustums()
        self.vis_params = []
        self.vis_index = 0
    def get_current_view_frustums(self,special_cam = None):
        self.activate_points,self.activate_frustrums = self.activate_group.get_view_frustums()
    def update_pose_all(self):
        if self.other_cg_list is None:
            return
        for ind, cg in enumerate(self.other_cg_list):
            relative_pose = self.relative_poses[ind+1]
            new_pose = np.dot(relative_pose, self.center_cam.pose)
            cg.reset_pose_all(new_pose)
    def reset_pose_all(self,pose):
        
        if self.activate_index == 0:
            center_pose = pose
        else:
            #start from 1 so do not +1
            relative_pose = self.relative_poses[self.activate_index]
            center_pose = np.linalg.inv(relative_pose) @ pose
        self.center_group.reset_pose_all(center_pose)
        self.update_pose_all()

    def update_activate_group(self,index=None):
        if self.other_cg_list is None:
            return
        if index is not None:
            self.activate_index = index
        self.activate_index = self.activate_index % len(self.all_cgs)
        self.update_pose_all()
        self.activate_group = self.all_cgs[self.activate_index]
        #print(f'now the activated group is {self.activate_index}')
    
    def save_global_vis_param(self,param):
        self.vis_param = param
    
    def append_global_vis_param_list(self,param):
        self.vis_params.append(param)

    def set_global_vis_param(self,index):
        self.vis_param = self.vis_params[index]

    def clear_rgbd(self):
        for cg in self.all_cgs:
            cg.clear_rgbd()
    
    
    
    def __getitem__(self,index):
        return self.all_cgs[index]
    def __len__(self):
        return len(self.all_cgs)
        





def red_check(img):
    # 读取图像
    image = img

    # 转换图像为LAB空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    plt.imshow(lab_image)
    plt.show()
    # 定义红色在LAB空间的范围
    lower_red = np.array([50, 150, 50])
    upper_red = np.array([255, 255, 255])

    # 创建一个掩码来提取红色区域
    red_mask = cv2.inRange(lab_image, lower_red, upper_red)

    # 使用轮廓检测来找到红色区域
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算红色区域的总面积
    red_area = 0
    for contour in contours:
        red_area += cv2.contourArea(contour)

    # 打印红色区域的面积
    print(f"红色区域的面积：{red_area} 像素")

    # 可以选择绘制红色区域的轮廓，如果需要
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Red Contours", red_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_camera(txt_dir,scale=1):
    if not os.path.exists(os.path.join(txt_dir,'images.txt')):
        bin2txt(txt_dir)
    img_txt = os.path.join(txt_dir,'images.txt')
    cam_txt = os.path.join(txt_dir, 'cameras.txt')
    with open(img_txt,'r') as f:
        ll = f.readlines()
    ll = ll[4:]

    ll = ll[::2]
    # target_name = target_name.split('.')[-2]
    cameras = []
    for l in ll:
        cont = l.split()
        rt = cont
        cam_ind = int(cont[8])
        #cam_name = os.path.basename(rt[9]).split('.')[0]
        cam_name = rt[9].split('.')[0]

        #QW, QX, QY, QZ = [float(rt[1]), float(rt[2]),  float(rt[3]),  float(rt[4])]
        Rq1 = np.asarray([float(rt[2]),  float(rt[3]),  float(rt[4]), float(rt[1])])
        r1 = R.from_quat(Rq1)
        Rm1 = r1.as_matrix()
        T = np.asarray([float(rt[5]), float(rt[6]), float(rt[7])]) * scale

        with open(cam_txt,'r') as f:
            ll = f.readlines()
            ll = ll[3:]
        K = ll[cam_ind - 1]
        K = K.split()[4:]
        K = [float(i) for i in K]
        K[0] = K[0]
        K[1] = K[1]
        cameras.append(camera(K,Rm1,T,cam_name))
    return cameras

def get_cam_set(txt_dir,scale = 1):
    if not os.path.exists(os.path.join(txt_dir,'images.txt')):
        bin2txt(txt_dir) 
    cameras_all = get_camera(txt_dir,scale)

    cam_group = {}

    for cam in cameras_all:
        #a big mistake! 1020-6.5!!!
        if cam.name == '1020-6':
            continue
        group_name = cam.name.split('/')[0]
        group_name = group_name.replace('1020-','')
        if group_name not in cam_group:
            cam_group[group_name] = []
        cam_group[group_name].append(cam)

    cam_set = [camera_group(cam_group[f'{i}']) for i in range(0,len(cam_group))]
    return cam_set