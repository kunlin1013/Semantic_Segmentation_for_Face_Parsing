from glob import glob
from pathlib import Path
import json
import numpy as np
from PIL import Image

def get_project_dir():

    """
    Get the full path to the repository
    """
    project_directory = r".\CelebAMask-HQ"
    
    return Path(project_directory)

data = [i for i in range(30000)]
data_json = {}
IMG_SIZE = (256, 256)
labels_celeb = ['background','skin','nose','eye_g','l_eye','r_eye','l_brow',
                'r_brow','l_ear','r_ear','mouth','u_lip','l_lip','hair','hat',
                'ear_r','neck_l','neck','cloth']

for i in data:
    print(i)
    f = glob(f'{get_project_dir()}\\CelebAMask-HQ-mask-anno\\{int(float(i) // 2000)}\\{int(i):05d}*.png')
    data_json[i] = {'filepath':f"{get_project_dir()}\\CelebA-HQ-img\\{i}.jpg", 
                    "label_path": f,
                    "npy_path": f"{get_project_dir()}\\Mask\\{i}.npy"}
    mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], len(labels_celeb)))
    BackGround = np.ones(IMG_SIZE)
    for mask_path in f:
        mask_image = np.array(Image.open(mask_path).resize(IMG_SIZE).convert('L')) # 轉成黑白影像
        label = mask_path.split('\\')[-1].replace('.png', '')[6:]
        BackGround[(mask_image == 255) & (BackGround == 1)] = 0
        mask[:,:,labels_celeb.index(label)] = np.where((mask_image > 128), 1, 0)
    mask[:,:,0] = BackGround
    np.save(rf'.\CelebAMask-HQ\Mask\{i}.npy', mask.astype("uint8"))
    
json_file_path = r'.\CelebAMask-HQ\data_dict.json'
with open(json_file_path, 'w') as json_file:
    # 在 JSON 文件中，indent=4 是一個用於格式化輸出的參數
    # 當使用 json.dump 或 json.dumps 方法時，設置 indent=4 會導致每個層級的數據以 4 個空格縮進，從而使輸出的 JSON 數據更易於閱讀
    json.dump(data_json, json_file, indent=4)