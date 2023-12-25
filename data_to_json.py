from glob import glob
from pathlib import Path
import json
from PIL import Image

def get_project_dir():

    """
    Get the full path to the repository
    """
    project_directory = r".\CelebAMask-HQ"
    
    return Path(project_directory)

data = glob(r".\unseen\*.jpg")
data_json = {}
IMG_SIZE = (1024, 1024)
labels_celeb = ['background','skin','nose','eye_g','l_eye','r_eye','l_brow',
                'r_brow','l_ear','r_ear','mouth','u_lip','l_lip','hair','hat',
                'ear_r','neck_l','neck','cloth']

txt = []
for i, path in enumerate(data):
    print(i)
    key = int(path.split('\\')[-1].replace('.jpg','').split('_')[0])
    data_json[key] = {'filepath':path, 
                    "npy_path": r".\CelebAMask-HQ\Mask\0.npy"}
    mask_image = Image.open(path).resize(IMG_SIZE).save(path)
    txt.append(key)
    
txt.sort()
    
json_file_path = r'.\unseen.json'
with open(json_file_path, 'w') as json_file:
    # 在 JSON 文件中，indent=4 是一個用於格式化輸出的參數
    # 當使用 json.dump 或 json.dumps 方法時，設置 indent=4 會導致每個層級的數據以 4 個空格縮進，從而使輸出的 JSON 數據更易於閱讀
    json.dump(data_json, json_file, indent=4)
    
txt_file_path = r'.\unseen.txt'
with open(txt_file_path, 'w') as file:
    for item in txt:
        file.write(f"{item}\n")