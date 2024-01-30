import numpy as np

import tensorflow as tf
import json
from CSV_process.csv_utils import array2csv

from segmentation_models.metrics import iou_score
from segmentation_models.losses import  bce_dice_loss
from Model_SegFormer.Attention import Attention
from Model_SegFormer.Head import MLP, ConvModule, SegFormerHead
from Model_SegFormer.modules import DWConv, Mlp, Block, OverlapPatchEmbed, MixVisionTransformer
from Model_SegFormer.utils import ResizeLayer, DropPath
from Load_Data import DataGenerator_test
from thresholding import convert_binary
import cv2
from scipy import ndimage

labels_celeb = ['background','skin','nose',
                'eye_g','l_eye','r_eye','l_brow',
                'r_brow','l_ear','r_ear','mouth',
                'u_lip','l_lip','hair','hat',
                'ear_r','neck_l','neck','cloth']


custom_objects = {"Attention" :  Attention,
                  "MLP" : MLP,
                  "ConvModule" : ConvModule,
                  "SegFormerHead" : SegFormerHead,
                  "DWConv" : DWConv,
                  "Mlp" : Mlp,
                  "Block" : Block,
                  "OverlapPatchEmbed" : OverlapPatchEmbed,
                  "MixVisionTransformer" : MixVisionTransformer,
                  "ResizeLayer" : ResizeLayer,
                  "DropPath" : DropPath,
                  "binary_crossentropy_plus_dice_loss" : bce_dice_loss,
                  "iou_score" : iou_score
                  }

BATCH_SIZE = 16
NUM_IMAGE_ITER = 300
IMG_SIZE = (256, 256)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def get_project_dir():
    # Get the full path to the repository
    return r".\CelebAMask-HQ"

if __name__ == '__main__':

    # Load testing data
    with open('unseen.txt', 'r') as file:
        data = file.read().split("\n")[:-1]
    test_list = data

    # Load json file
    json_file_path = "unseen.json"
    with open(json_file_path, 'r') as json_file:
        data_json = json.load(json_file)

    # Load trained model
    net_final = tf.keras.models.load_model(r'.\Weight\Segformer\8729-26-0.241.h5',custom_objects=custom_objects)

    # Iteratively predict masks
    num_total_image = len(test_list)
    num_iter = int(np.ceil(num_total_image / NUM_IMAGE_ITER))
    is_first_row = True

    for it in range(num_iter):
        starting_image_id = it * NUM_IMAGE_ITER
    
        # Fetch test data and predict result
        test, test_count = DataGenerator_test(dir='test',
                                              img_list=test_list[starting_image_id:np.min((starting_image_id+NUM_IMAGE_ITER, num_total_image))],
                                              data_json=data_json,
                                              IsAugmentation=False,
                                              batch_size=BATCH_SIZE)
        
        predict_test = net_final.predict(test, verbose=1)
    
        # Threshold using fixed value
        predict_result = convert_binary(predict_test, 0.1)
        
        # for num in range(test_count):
        #     img = predict_result[num,:,:,:]
    
        #     # BackGround = np.ones(IMG_SIZE).astype("uint8") * 255
        #     for idx in range(1, img.shape[-1]):
        #         mask = img[:, :, idx]                
        #         if labels_celeb[idx] in ["nose", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip"]:
        #             if np.max(mask) == 255:
        #                 mask_binary = (mask > 0).astype(int)        # convert to binary image 
        #                 structure =  np.ones((3, 3), np.uint8)
        #                 labeled_array, num_features = ndimage.label(mask_binary, structure=structure)   # Connected Component Labeling
        #                 image_center = np.array([mask_binary.shape[0] // 2, mask_binary.shape[1] // 2]) # Find the center of image
        #                 centers_of_mass = ndimage.center_of_mass(mask_binary, labeled_array, range(1, num_features + 1)) # Find the center of each component
        #                 repeat = []
        #                 for i, center1 in enumerate(centers_of_mass):
        #                     for j, center2 in enumerate(centers_of_mass):
        #                         if i != j and j > i: 
        #                             distance = np.linalg.norm(np.array(center1) - np.array(center2))
        #                             # If the components are closer than the threshold, merge them
        #                             if distance < 30:
        #                                 repeat.append(center2)
        #                                 num_features -= 1
        #                                 labeled_array[labeled_array == j+1] = i + 1
        #                 centers_of_mass = [element for element in centers_of_mass if element not in repeat]
        #                 if num_features >= 2:
        #                     distances_to_center = [np.linalg.norm(np.array(center) - image_center) for center in centers_of_mass]
        #                     Not_central_label = np.argmax(distances_to_center) + 1  # +1 because labels start from 1
        #                     mask = np.where(labeled_array == Not_central_label, 0, mask)
    
        #         if labels_celeb[idx] in ["hair", "cloth", "skin", "hat"]:
        #             kernel = np.ones((3,3), np.uint8)
        #             opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #             mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) 
        #         # BackGround[(mask == 255) & (BackGround == 255)] = 0
        #         predict_result[num,:,:,idx] = mask
        #     # predict_result[num,:,:,0] = BackGround
        
        # Convert prediction to .csv fiile
        array2csv(predict_result, starting_image_id=starting_image_id, header_needed=is_first_row)
        is_first_row = False
#%%

