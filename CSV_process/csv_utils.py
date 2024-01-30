# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']
def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if type(img) is type(None):
        return np.zeros((256, 256, 1), dtype=np.uint8)
    return img

def mask2binary(path):
    mask = read_mask(path)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0,1,0)
    return mask

def rle_encode(img): 
    pixels = img.flatten()
    if np.sum(pixels)==0:
        return '0'
    pixels = np.concatenate([[0], pixels, [0]]) 
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 
    runs[1::2] -= runs[::2]
    # to string sep='_'
    runs = '_'.join(str(x) for x in runs)
    return runs

def rle_decode(mask_rle, shape): 
    s = mask_rle.split('_')
    s = [0 if x=='' else int(x) for x in s]
    if np.sum(s)==0:
        return np.zeros(shape, dtype=np.uint8)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])] 
    starts -= 1 
    ends = starts + lengths 
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8) 
    for lo, hi in zip(starts, ends): 
         img[lo:hi] = 255
    return img.reshape(shape)

def mask2csv(mask_paths, csv_path='mask.csv',image_id=1,header=False):
    """
        mask_paths: dict of label:mask_paths
        ['label1':path1,'label2':path2,...]
    """
    results = []
    for i, label in enumerate(labels_celeb):
        try:
            mask = mask2binary(mask_paths[label])
        except:
            mask = np.zeros((256, 256), dtype=np.uint8)
        mask = rle_encode(mask)
        results.append(mask)
    df = pd.DataFrame(results)
    df.insert(0,'label',labels_celeb)
    # df.insert(0,'Usage',["Public" for i in range(len(results))])
    df.insert(0,'ID',[image_id*19+i for i in range(19)])
    if header:
        df.columns = ['ID','label','segmentation']
    # print()
    print(df)
    df.to_csv(csv_path,mode='a',header=header,index=False)

def mask2csv2(masks, csv_path='mask.csv',image_id=1,header=False):
    """
        mask: matrix with size (H, W, 19)
    """
    results = []
    for i, label in enumerate(labels_celeb):
        try:
            mask = masks[i]
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        except:
            mask = np.zeros((256, 256), dtype=np.uint8)
        mask = rle_encode(mask)
        results.append(mask)
    df = pd.DataFrame(results)
    df.insert(0,'label',labels_celeb)
    # df.insert(0,'Usage',["Public" for i in range(len(results))])
    df.insert(0,'ID',[image_id*19+i for i in range(19)])
    
    if header:
        df.columns = ['ID','label','segmentation']
    # print()
    # print(df)
    df.to_csv(csv_path,mode='a',header=header,index=False)

def array2csv(mask_array, csv_path='mask.csv', starting_image_id=0, header_needed=False):
    '''
        mask_array: Result of the predicted mask array with shape (num_test_img, H, W, 19)
    '''
    # Dictionary of mask path to pass to mask2csv function
    mask_array = mask_array.transpose((0, 3, 1, 2))
    header_needed = header_needed
    for id in range(starting_image_id, starting_image_id + mask_array.shape[0]):
        mask2csv2(mask_array[id-starting_image_id], image_id=id, header=header_needed)
        header_needed = False
    
if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import os

    # Determine the path of the results
    result_path = '.'
    img_id = [0, 1]

    # # Using mask2csv
    # # Create random mask and save file
    # for id in img_id:
    #     id_path = os.path.join(result_path, str(id))
    #     if not os.path.isdir(id_path):
    #         os.mkdir(id_path)
    #     for label in labels_celeb:
    #         if label == 'background':
    #             mask = np.zeros((256, 256), dtype=np.uint8)
    #         else:
    #             mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
    #         mask = Image.fromarray(mask)
    #         mask_path = os.path.join(id_path, label + '.png')
    #         mask.save(mask_path)

    # # Dictionary of mask path to pass to mask2csv function
    # header_needed = True
    # for id in range(len(img_id)):
    #     mask_path = {label:os.path.join(str(id), label + '.png') for label in labels_celeb}
    #     mask2csv(mask_path, image_id=id, header=header_needed)
    #     header_needed = False

    # Using mask2csv2
    # Create random mask and save file
    mask = np.zeros((len(img_id), 256, 256, len(labels_celeb)))
    for i, _ in enumerate(img_id):
        background = np.zeros((256, 256, 1), dtype=np.uint8)
        mask[i] = np.concatenate((background,
                               np.random.randint(0, 2, (256, 256, 18), dtype=np.uint8) * 255),
                               axis=-1)

    # # Dictionary of mask path to pass to mask2csv function
    # mask = mask.transpose((0, 3, 1, 2))
    # header_needed = True
    # for id in range(len(img_id)):
    #     mask2csv2(mask[id], image_id=id, header=header_needed)
    #     header_needed = False
        
    array2csv(mask, starting_image_id=0, header_needed=False)