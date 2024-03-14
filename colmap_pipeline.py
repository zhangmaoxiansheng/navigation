import os
import subprocess
import ray
from bin2txt import *
from ply_io import *
from natsort import natsorted
#@ray.remote
def process_cpu(proj_path, img_path):
    sparse_path = os.path.join(proj_path, 'sparse')
    os.makedirs(sparse_path, exist_ok=True)
    
    # 使用 subprocess.run 替换 os.system
    subprocess.run(['colmap', 'feature_extractor', '--database_path', f'{proj_path}/ddd.db', '--image_path', img_path,'--ImageReader.camera_model','PINHOLE','--SiftExtraction.use_gpu=false'])
    subprocess.run(['colmap', 'exhaustive_matcher', '--database_path', f'{proj_path}/ddd.db','--SiftMatching.use_gpu=false'])
    # 使用 subprocess.run 替换 os.system
    subprocess.run(['colmap', 'mapper', '--database_path', f'{proj_path}/ddd.db', '--image_path', img_path, '--output_path', sparse_path])
    model_path = os.path.join(sparse_path,'0')
    bin2txt(model_path)
    ply_io(model_path)
    return model_path
#@ray.remote(num_cpus=1, num_gpus=1)
def process(proj_path, img_path):
    sparse_path = os.path.join(proj_path,'sparse')
    if os.path.exists(sparse_path):
        os.system(f'rm -r {sparse_path}')
    os.makedirs(sparse_path,exist_ok=True)
    os.system(f'colmap feature_extractor --database_path {proj_path}/ddd.db --image_path {img_path} --ImageReader.camera_model PINHOLE > {proj_path}/col_output.txt')
    os.system(f'colmap exhaustive_matcher --database_path {proj_path}/ddd.db > {proj_path}/col_output.txt')
    
    os.system(f'colmap mapper --database_path {proj_path}/ddd.db --image_path {img_path} --output_path {sparse_path} > {proj_path}/col_output.txt')
    model_path = os.path.join(sparse_path,'0')
    if len(os.listdir(sparse_path))>1:
        ll = os.listdir(sparse_path)
        ll = natsorted(ll)
        ll = [os.path.join(sparse_path,l) for l in ll]
        last = ll[:-1]
        for i in ll[:-1]:
            os.system(f'rm -r {i}')
        os.system(f'mv {last} {model_path}')
    # if len(os.listdir(sparse_path)) == 1:
    #     ll = os.listdir(sparse_path)[0]
    #     ll = os.path.join(sparse_path,ll)
    #     os.system(f'mv {ll} {model_path}')
    
    bin2txt(model_path)
    ply_io(model_path)
    return model_path



if __name__ == '__main__':
    # proj_path = './sim_test/0_6'
    # img_path = './sim_test/0_6/images'
    #ray.get(process.remote(proj_path,img_path))
    #process(proj_path,img_path)
    #process(proj_path,img_path)
    #res = colmap_process.remote(proj_path,img_path)
    #ray.get(colmap_process.remote(proj_path,img_path))
    #print('main process is alive')
    #res_value = ray.get(res)
    root = sys.argv[1]
    #root = './capture_test/bridge'
    for d in os.listdir(root):
        proj_path = os.path.join(root,d)
        img_path = os.path.join(proj_path,'images')
        process(proj_path,img_path)
        print(proj_path)
