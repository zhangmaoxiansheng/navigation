from navigation import *

def random_init(navi):
    move_key = ['W','S','A','D','Q','E']
    rotate_key = ['U','O']
    keys = [random.choice(move_key),random.choice(rotate_key)]

    if keys[0] == 'W' or keys[0] == 'S':
        step = random.uniform(0,0.1)
    else:
        step = random.uniform(0,0.3)

    rotate_step = random.uniform(0, math.pi / 6)
    #rotate_step = 0
    #step = 0.2*5
    navi.set_step(step,rotate_step)
    
    print(f'move action:{keys[0]}, step:{step}')
    print(f'rotate action:{keys[1]},step:{rotate_step}')
    navi.replay(keys=keys,only_move=True)
    navi.reset_vis()

def annotation(sample_dir, model_dir, pcd_dir, num):
    cam_all = get_camera(model_dir)
    pcd = o3d.io.read_point_cloud(pcd_dir)
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.1)
    frame_num_list = [47,49,50,51,52,53]#48 54 is not good
    vis_params = []
    for i in range(num):
        frame_num = random.choice(frame_num_list)
        print(frame_num)
        #frame_num = 47
        frame_name = f'frame_{frame_num}_'
        cams = []
        for cam in cam_all:
            if frame_name in cam.name:
                cams.append(copy.deepcopy(cam))
        cams = camera_group(cams,center_index='7',special_index='1')

        cam_set = camera_set([cams])
        cam_set.vis_params = vis_params
        if vis_params:
            cam_set.set_global_vis_param(0)

        save_dir = os.path.join(sample_dir,f'sample-{i}')
        os.makedirs(save_dir,exist_ok=True)
        recorder = os.path.join(save_dir,'record.txt')
        navi = navigator(cam_set,voxel,txt=recorder, save_dir=save_dir)
        random_init(navi)
        navi.run()
        vis_params = copy.deepcopy(cam_set.vis_params)
        #open3d bug
        del navi




if __name__ == '__main__':
    model_dir = './20240306_3/sparse/0'
    pcd_dir = os.path.join(model_dir,'fused-crop.ply')

    sample_dir = '20240306-samples'
    sample_num = 50
    annotation(sample_dir,model_dir,pcd_dir,sample_num)
    