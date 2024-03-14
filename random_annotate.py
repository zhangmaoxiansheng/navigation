import os
import random
import math
from tqdm import tqdm
import numpy as np


def random_generator(extrinsic,step_size=0.2,rotate_step_size=0.02,random_step=False,random_rotate=False,no_ws = False):
    if no_ws:
        keys = list('ADQEJKLUIO') 
    else:
        keys = list('WASDQEJKLUIO')
    keys = [ord(k) for k in keys]
    if random_step:
        keys = keys[:6]
    if random_rotate:
        keys = keys[6:]
    key = random.choice(keys)
    if key == 'W' or key =='S':
        step_size = min(step_size,0.1)
    
    forward_vector = extrinsic[:3,:3].T@extrinsic[:3, 2]
    right_vector = extrinsic[:3,:3].T@extrinsic[:3, 0]
    up_vector = extrinsic[:3,:3].T@extrinsic[:3, 1]

    if key == ord('W'):
        extrinsic[0:3, 3] -= (step_size * forward_vector)
    if key == ord('S'):
        extrinsic[0:3, 3] += (step_size * forward_vector)
    if key == ord('A'):
        extrinsic[0:3, 3] += (step_size * right_vector)
    if key == ord('D'):
        extrinsic[0:3, 3] -= (step_size * right_vector)
    if key == ord('Q'):
        extrinsic[0:3, 3] += (step_size * up_vector)
    if key == ord('E'):
        extrinsic[0:3, 3] -= (step_size * up_vector)
    if key == ord('U'):
        rotation_matrix = np.array([[np.cos(rotate_step_size), -np.sin(rotate_step_size), 0],[np.sin(rotate_step_size), np.cos(rotate_step_size), 0],[0, 0, 1]])
        extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
        extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
    if key == ord('O'):
        rotation_matrix = np.array([[np.cos(-rotate_step_size), -np.sin(-rotate_step_size), 0],[np.sin(-rotate_step_size), np.cos(-rotate_step_size), 0],[0, 0, 1]])
        extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
        extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
    if key == ord('J'):
    # 绕Y轴旋转
        rotation_matrix = np.array([[np.cos(rotate_step_size), 0, np.sin(rotate_step_size)],[0, 1, 0],[-np.sin(rotate_step_size), 0, np.cos(rotate_step_size)]])
        extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
        extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
    if key == ord('L'):
        # 绕Y轴逆时针旋转
        rotation_matrix = np.array([[np.cos(-rotate_step_size), 0, np.sin(-rotate_step_size)],[0, 1, 0],[-np.sin(-rotate_step_size), 0, np.cos(-rotate_step_size)]])
        extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
        extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
    if key == ord('K'):
        # 绕X轴旋转
        rotation_matrix = np.array([[1, 0, 0],[0, np.cos(rotate_step_size), -np.sin(rotate_step_size)],[0, np.sin(rotate_step_size), np.cos(rotate_step_size)]])
        extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
        extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
    if key == ord('I'):
        # 绕X轴逆时针旋转
        rotation_matrix = np.array([[1, 0, 0],[0, np.cos(-rotate_step_size), -np.sin(-rotate_step_size)],[0, np.sin(-rotate_step_size), np.cos(-rotate_step_size)]])
        extrinsic[:3, :3] = rotation_matrix.dot(extrinsic[:3, :3])
        extrinsic[:3, 3] = rotation_matrix.dot(extrinsic[:3, 3])
    return extrinsic


sample_root = './20240306-samples-new'

example_samples = list(range(20,50))

for i in tqdm(range(50,150)):
    sample_num = random.choice(example_samples)
    example = os.path.join(sample_root,f'sample-{sample_num}')
    sample = os.path.join(sample_root,f'sample-{i}')
    print(example)
    print(sample)
    os.system(f'cp -r {example} {sample}')
    for step_dir in os.listdir(sample):
        print(step_dir)
        step = int(step_dir)
        if step > len(os.listdir(sample))-3:
            continue
        if step > 6:
            continue

        RT = os.path.join(sample,step_dir,'RT')
        RT_new = os.path.join(sample,step_dir,'RT2')
        os.makedirs(RT_new,exist_ok=True)
        
        if step == 0:
            step = random.uniform(0,0.8)
            rotate_step = random.uniform(0, math.pi / 12)
        elif step == 1:
            step = random.uniform(0,0.3)
            rotate_step = random.uniform(0, 0.04)
        else:
            step = random.uniform(0,0.2)
            rotate_step = random.uniform(0, 0.02)


        for iii in range(13):
            R = np.load(os.path.join(RT,f'camera_{iii}_R.npy'))
            T = np.load(os.path.join(RT,f'camera_{iii}_T.npy'))
            pose = np.zeros((4,4))
            pose[:3,:3] = R
            pose[:3, 3] = T

            if step == 0:
                pose = random_generator(pose,step_size=step,rotate_step_size=rotate_step,random_step=True)
                pose = random_generator(pose,step_size=step,rotate_step_size=rotate_step,random_rotate=True)
            elif step == 1:
                pose = random_generator(pose,step_size=step,rotate_step_size=rotate_step,random_step=True,no_ws=True)
                pose = random_generator(pose,step_size=step,rotate_step_size=rotate_step,random_rotate=True)
            else:
                pose = random_generator(pose,step_size=step,rotate_step_size=rotate_step,no_ws=True)

            R = pose[:3,:3]
            T = pose[:3, 3]
            np.save(os.path.join(RT_new,f'camera_{iii}_R.npy'),R)
            np.save(os.path.join(RT_new,f'camera_{iii}_T.npy'),T)
            
        os.system(f'rm -r {RT}')
        os.system(f'mv {RT_new} {RT}')


    